import numpy as np

from codec import A2process, blocking, quantization, evaluation, A3process_parallel
from codec.decoder import entropy_decode, quantization_decode, transform_decode, differential_decode, prediction_decode
from codec.encoder import entropy_encode, prediction_encode_row, differential_encode, prediction_encode, \
    transform_encode, quantization_encode
from utils import reader


def search_qp(target, table):
    left = 0
    right = len(table) - 1
    while left < right:
        mid = (right + left) // 2
        # If x is greater, ignore left half
        if table[mid] > target:
            left = mid + 1
        # If x is smaller, ignore right half
        elif table[mid] < target:
            right = mid
        # means x is present at mid
        else:
            return mid
    # If we reach here, then the element was not present
    return right


def encode_complete(filepath, config_dict, table_dict):
    w = config_dict['w']
    h = config_dict['h']
    i = config_dict['i']
    n = config_dict['n']
    r = config_dict['r']
    qp = config_dict['qp']
    period = config_dict['period']
    frame = config_dict['frame']
    nRefFrames = config_dict['nRefFrames']
    VBSEnable = config_dict['VBSEnable']
    # this is the coefficient to adjust lambda value
    lambda_coefficient = config_dict['lambda_coefficient']
    FMEEnable = config_dict['FMEEnable']
    FastME = config_dict['FastME']
    RCFlag = config_dict['RCFlag']
    targetBR = config_dict['targetBR']
    fps = config_dict['fps']
    intraLine = config_dict['intraLine']
    ParallelMode = config_dict['ParallelMode']
    bits_tables = {}
    if w == 352 and h == 288:
        bits_tables['intra'] = table_dict['CIF_intra']
        bits_tables['inter'] = table_dict['CIF_inter']
    elif w == 176 and h == 144:
        bits_tables['intra'] = table_dict['QCIF_intra']
        bits_tables['inter'] = table_dict['QCIF_inter']
    if RCFlag == 0 and ParallelMode == 0:
        A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                                  FMEEnable, FastME, frame)
    elif ParallelMode in [1, 2]:
        A3process_parallel.encode_parallel_1_2(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                                  FMEEnable, FastME, ParallelMode, frame)
    elif ParallelMode == 3:
        A3process_parallel.encode_parallel_3(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                                  FMEEnable, FastME, ParallelMode, frame)
    elif RCFlag == 1:
        encode_RC_1(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable, FastME,
                    RCFlag, targetBR, fps, bits_tables, frame)
    elif RCFlag in [2, 3]:
        encode_RC_2(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable, FastME,
                    RCFlag, targetBR, fps, bits_tables, intraLine, frame)


def encode_RC_1(filepath, w, h, block_size, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable,
                FastME, RCFlag, targetBR, fps, bits_tables, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, block_size, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    line, bits = entropy_encode.entropy_encode_setting(w, h, block_size, qp, period, VBSEnable, FMEEnable, RCFlag)
    diff_file.write(line)
    diff_file.write("\n")
    bit_count_arr = []
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction_array = []
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    bits_budget_frame = targetBR / fps
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        bit_sum = 0
        budget_curr_frame = bits_budget_frame
        if x % period == 0:
            prediction = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)
            for i in range(n_rows_frame):
                budget_per_row = budget_curr_frame / (n_rows_frame - i)
                qp = search_qp(budget_per_row, bits_tables['intra'])
                lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
                q = quantization.generate_q(block_size, qp)
                q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
                prediction, vec, split, res_code = prediction_encode_row.intra_residual_row(frame_block_array[x], n,
                                                                                            lambda_val, q, q_split,
                                                                                            VBSEnable, prediction, i)
                residual_file.write(res_code)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                diff_file.write(code)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    diff_file.write(code)
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_row += bit_count
                # update budget
                budget_curr_frame -= bit_row
                bit_sum += bit_row
            prediction = blocking.deblock_frame(prediction, w, h)
            prediction_array = [prediction]
        else:
            block_itran = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
            vec_array = []
            split_array = []
            for i in range(n_rows_frame):
                budget_per_row = budget_curr_frame / (n_rows_frame - i)
                qp = search_qp(budget_per_row, bits_tables['inter'])
                lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
                q = quantization.generate_q(block_size, qp)
                q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
                block_itran, vec, split, res_code = prediction_encode_row.generate_residual_ME_row(prediction_array,
                                                                                                   frame_block_array[x],
                                                                                                   w,
                                                                                                   h, n, r, lambda_val,
                                                                                                   q, q_split,
                                                                                                   FMEEnable,
                                                                                                   FastME, VBSEnable,
                                                                                                   block_itran, i)
                residual_file.write(res_code)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                diff_file.write(code)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    diff_file.write(code)
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_row += bit_count
                # update budget
                budget_curr_frame -= bit_row
                bit_sum += bit_row
                # for reconstruction
                vec_array += vec
                split_array += split
            res = blocking.deblock_frame(block_itran)
            prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array, split_array, w, h,
                                                                  block_size,
                                                                  FMEEnable)
            prediction_array.insert(0, prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        bit_count_arr.append(bit_sum)
    residual_file.close()
    diff_file.close()


def encode_RC_2(filepath, w, h, block_size, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable,
                FastME, RCFlag, targetBR, fps, bits_tables, intraLine, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, block_size, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    line, bits = entropy_encode.entropy_encode_setting(w, h, block_size, qp, period, VBSEnable, FMEEnable, RCFlag)
    diff_file.write(line)
    diff_file.write("\n")
    bit_count_arr = []
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction_array = []
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    # first run
    intra_profile = []
    budget_profile = []
    vec_arr_first = []
    split_arr_first = []
    print('starting first pass')
    lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
    q = quantization.generate_q(block_size, qp)
    q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        bit_sum = 0
        bit_by_row = []
        vec_each_row = []
        split_each_row = []
        if x % period == 0:
            prediction = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)
            for i in range(n_rows_frame):
                prediction, vec, split, res_code = prediction_encode_row.intra_residual_row(frame_block_array[x], n,
                                                                                            lambda_val, q, q_split,
                                                                                            VBSEnable, prediction, i)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                bit_row += bit_count
                vec_each_row.append(vec)
                split_each_row.append(split)
                # update budget
                bit_sum += bit_row
                bit_by_row.append(bit_row)
            prediction = blocking.deblock_frame(prediction, w, h)
            prediction_array = [prediction]
        else:
            block_itran = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
            vec_array = []
            split_array = []
            for i in range(n_rows_frame):
                block_itran, vec, split, res_code = prediction_encode_row.generate_residual_ME_row(prediction_array,
                                                                                                   frame_block_array[x],
                                                                                                   w,
                                                                                                   h, n, r, lambda_val,
                                                                                                   q, q_split,
                                                                                                   FMEEnable,
                                                                                                   FastME, VBSEnable,
                                                                                                   block_itran, i)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                bit_row += bit_count
                vec_each_row.append(vec)
                split_each_row.append(split)
                # update budget
                bit_sum += bit_row
                bit_by_row.append(bit_row)
                # for reconstruction
                vec_array += vec
                split_array += split
            res = blocking.deblock_frame(block_itran)
            prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array, split_array, w, h,
                                                                  block_size,
                                                                  FMEEnable)
            prediction_array.insert(0, prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        bit_count_arr.append(bit_sum)
        vec_arr_first.append(vec_each_row)
        split_arr_first.append(split_each_row)
        # bit_by_row_arr.append(bit_by_row)
        # process bit results
        if bit_sum / n_rows_frame > intraLine * bits_tables['inter'][qp] or x % period == 0:
            intra_profile.append(True)
        else:
            intra_profile.append(False)
        budget_row = []
        for each_row in bit_by_row:
            budget_row.append(each_row / bit_sum)
        budget_profile.append(budget_row)

    # find scaling of table
    I_frame_count = 0
    I_frame_sum = 0
    P_frame_count = 0
    P_frame_sum = 0
    for x in range(num_frames):
        if x % period == 0:
            I_frame_count += 1
            I_frame_sum += bit_count_arr[x] / n_rows_frame
        elif not intra_profile[x]:
            P_frame_count += 1
            P_frame_sum += bit_count_arr[x] / n_rows_frame
    I_scale = 1
    if I_frame_sum > 0:
        I_frame_sum /= I_frame_count
        I_scale = I_frame_sum / bits_tables['intra'][qp]
    P_scale = 1
    if P_frame_sum > 0:
        P_frame_sum /= P_frame_count
        P_scale = P_frame_sum / bits_tables['inter'][qp]

    for qp_val in range(12):
        bits_tables['intra'][qp_val] = int(bits_tables['intra'][qp_val] * I_scale)
        bits_tables['inter'][qp_val] = int(bits_tables['inter'][qp_val] * P_scale)

    print(intra_profile)

    # second run
    print('starting second pass')
    bit_count_arr = []
    bits_budget_frame = targetBR / fps
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        bit_sum = 0
        budget_curr_frame = bits_budget_frame
        budget_percent_remain = sum(budget_profile[x])
        vec_row_profile = vec_arr_first[x]
        split_row_profile = split_arr_first[x]
        if intra_profile[x]:
            # intra indicator in diff file first (1)
            code, bit_count = entropy_encode.exp_golomb(1)
            diff_file.write(code)
            bit_sum += bit_count
            prediction = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)
            for i in range(n_rows_frame):
                budget_per_row = int(budget_curr_frame * budget_profile[x][i] / budget_percent_remain)
                qp = search_qp(budget_per_row, bits_tables['intra'])
                lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
                q = quantization.generate_q(block_size, qp)
                q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
                if RCFlag == 2:
                    prediction, vec, split, res_code = prediction_encode_row.intra_residual_row(frame_block_array[x], n,
                                                                                            lambda_val, q, q_split,
                                                                                            VBSEnable, prediction, i)
                elif RCFlag == 3:
                    prediction, vec, split, res_code = prediction_encode_row.intra_residual_row_leverage(frame_block_array[x], n,
                                                                                                q, q_split,
                                                                                                prediction,
                                                                                                i, split_row_profile[i])
                residual_file.write(res_code)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                diff_file.write(code)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    diff_file.write(code)
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_row += bit_count
                # update budget
                budget_percent_remain -= budget_profile[x][i]
                budget_curr_frame -= bit_row
                bit_sum += bit_row
            prediction = blocking.deblock_frame(prediction, w, h)
            prediction_array = [prediction]
        else:
            # inter indicator in diff file first (0)
            code, bit_count = entropy_encode.exp_golomb(0)
            diff_file.write(code)
            bit_sum += bit_count
            block_itran = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
            vec_array = []
            split_array = []
            for i in range(n_rows_frame):
                budget_per_row = int(budget_curr_frame * budget_profile[x][i] / budget_percent_remain)
                qp = search_qp(budget_per_row, bits_tables['inter'])
                lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
                q = quantization.generate_q(block_size, qp)
                q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
                if RCFlag == 2:
                    block_itran, vec, split, res_code = prediction_encode_row.generate_residual_ME_row(prediction_array,
                                                                                                   frame_block_array[x],
                                                                                                   w,
                                                                                                   h, n, r, lambda_val,
                                                                                                   q, q_split,
                                                                                                   FMEEnable,
                                                                                                   FastME, VBSEnable,
                                                                                                   block_itran, i)
                elif RCFlag == 3:
                    block_itran, vec, split, res_code = prediction_encode_row.generate_residual_ME_row_leverage(prediction_array,
                                                                                                       frame_block_array[
                                                                                                           x],
                                                                                                       w,
                                                                                                       h, n, r,
                                                                                                       q, q_split,
                                                                                                       FMEEnable,
                                                                                                       block_itran, i, split_row_profile[i], vec_row_profile[i])
                residual_file.write(res_code)
                bit_row = len(res_code)
                # diff file will be qp, split (if needed), mode or vec
                code, bit_count = entropy_encode.exp_golomb(qp)
                diff_file.write(code)
                bit_row += bit_count
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    diff_file.write(code)
                    bit_row += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_row += bit_count
                # update budget
                budget_percent_remain -= budget_profile[x][i]
                budget_curr_frame -= bit_row
                bit_sum += bit_row
                # for reconstruction
                vec_array += vec
                split_array += split
            res = blocking.deblock_frame(block_itran)
            prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array, split_array, w, h,
                                                                  block_size,
                                                                  FMEEnable)
            prediction_array.insert(0, prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        bit_count_arr.append(bit_sum)
    residual_file.close()
    diff_file.close()
    # np.savetxt(filepath + 'bits.txt', bit_count_arr, '%d')


def decode_complete(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
    w, h, i, qp, period, VBSEnable, FMEEnable, RCFlag, ParallelMode = entropy_decode.decode_setting(setting)
    if RCFlag == 0 and ParallelMode == 0:
        A2process.decode_complete(filepath)
    elif ParallelMode == 1:
        A3process_parallel.decode_parallel_1(filepath)
    elif ParallelMode in [2, 3]:
        A3process_parallel.decode_parallel_2(filepath)
    elif RCFlag == 1:
        decode_RC_1(filepath)
    elif RCFlag in [2, 3]:
        decode_RC_2(filepath)


def decode_RC_1(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, block_size, qp, period, VBSEnable, FMEEnable, RCFlag, ParallelMode = entropy_decode.decode_setting(setting)
    video = []
    recon_array = []
    n_rows_frame = (h - 1) // block_size + 1  # n_block_h
    n_cols_frame = (w - 1) // block_size + 1  # n_block_w
    frame_count = 0
    while len(res_code) > 0 and len(vec_code) > 0:
        print("decode frame: " + str(frame_count))
        quan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size))
        dequan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
        if frame_count % period == 0:
            vec_array = np.empty(0, dtype=np.uint16)  # vector or mode
        else:
            vec_array = np.empty((0, 3), dtype=np.uint16)  # vector or mode
        split_array = np.empty(0, dtype=np.uint16)
        for i in range(n_rows_frame):
            qp, vec_code = entropy_decode.get_num(vec_code, 1)
            qp = qp[0]
            q = quantization.generate_q(block_size, qp)
            q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
            if VBSEnable:
                # get the row of split indicators
                split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_cols_frame)
                split_row = differential_decode.differential_decode(split_diff)
            else:
                split_row = np.zeros(n_cols_frame)
            array_len = len(split_row) + 3 * np.sum(split_row)
            quan_frame, res_code = entropy_decode.decode_quan_frame_VBS_given_row(res_code, n_cols_frame, block_size,
                                                                                  split_row, quan_frame, i)
            dequan_frame = quantization_decode.dequantization_frame_VBS_given_row(quan_frame, q, q_split, split_row,
                                                                                  dequan_frame, i)
            # extract mode array or vector array
            if frame_count % period == 0:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame_alter(vec_code, array_len, False)
                vec_row = differential_decode.differential_decode(vec_diff)
                vec_array = np.append(vec_array, vec_row)
            else:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame_alter(vec_code, array_len, True)
                vec_row = differential_decode.differential_decode(vec_diff)
                vec_array = np.append(vec_array, vec_row, axis=0)
            split_array = np.append(split_array, split_row)
        itran = transform_decode.inverse_transform_frame_VBS(dequan_frame, split_array)
        res = blocking.deblock_frame(itran, w, h)
        if frame_count % period == 0:
            recon = prediction_decode.intra_decode_VBS(res, vec_array, split_array, w, h, block_size)
            recon_array = [recon]
        else:
            recon = prediction_decode.decode_residual_ME_VBS(recon_array, res, vec_array, split_array, w, h, block_size,
                                                             FMEEnable)
            recon_array.insert(0, recon)
            # here use the upper limit of nRefFrame for easier coding purpose
            if len(recon_array) == 4:
                recon_array = recon_array[:4]
        video.append(recon)
        frame_count += 1
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')


def decode_RC_2(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, block_size, qp, period, VBSEnable, FMEEnable, RCFlag, ParallelMode = entropy_decode.decode_setting(setting)
    video = []
    recon_array = []
    n_rows_frame = (h - 1) // block_size + 1  # n_block_h
    n_cols_frame = (w - 1) // block_size + 1  # n_block_w
    frame_count = 0
    while len(res_code) > 0 and len(vec_code) > 0:
        print("decode frame: " + str(frame_count))
        quan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size))
        dequan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
        intra_indicate, vec_code = entropy_decode.get_num(vec_code, 1)
        intra_indicate = intra_indicate[0]
        if intra_indicate == 1:
            vec_array = np.empty(0, dtype=np.uint16)  # vector or mode
        else:
            vec_array = np.empty((0, 3), dtype=np.uint16)  # vector or mode
        split_array = np.empty(0, dtype=np.uint16)
        for i in range(n_rows_frame):
            qp, vec_code = entropy_decode.get_num(vec_code, 1)
            qp = qp[0]
            q = quantization.generate_q(block_size, qp)
            q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
            if VBSEnable:
                # get the row of split indicators
                split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_cols_frame)
                split_row = differential_decode.differential_decode(split_diff)
            else:
                split_row = np.zeros(n_cols_frame)
            array_len = len(split_row) + 3 * np.sum(split_row)
            quan_frame, res_code = entropy_decode.decode_quan_frame_VBS_given_row(res_code, n_cols_frame, block_size,
                                                                                  split_row, quan_frame, i)
            dequan_frame = quantization_decode.dequantization_frame_VBS_given_row(quan_frame, q, q_split, split_row,
                                                                                  dequan_frame, i)
            # extract mode array or vector array
            if intra_indicate == 1:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame_alter(vec_code, array_len, False)
                vec_row = differential_decode.differential_decode(vec_diff)
                vec_array = np.append(vec_array, vec_row)
            else:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame_alter(vec_code, array_len, True)
                vec_row = differential_decode.differential_decode(vec_diff)
                vec_array = np.append(vec_array, vec_row, axis=0)
            split_array = np.append(split_array, split_row)
        itran = transform_decode.inverse_transform_frame_VBS(dequan_frame, split_array)
        res = blocking.deblock_frame(itran, w, h)
        if intra_indicate == 1:
            recon = prediction_decode.intra_decode_VBS(res, vec_array, split_array, w, h, block_size)
            recon_array = [recon]
        else:
            recon = prediction_decode.decode_residual_ME_VBS(recon_array, res, vec_array, split_array, w, h, block_size,
                                                             FMEEnable)
            recon_array.insert(0, recon)
            # here use the upper limit of nRefFrame for easier coding purpose
            if len(recon_array) == 4:
                recon_array = recon_array[:4]
        video.append(recon)
        frame_count += 1
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')