import numpy as np

from codec import A2process, blocking, quantization, evaluation
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
    bits_tables = {}
    if w == 352 and h == 288:
        bits_tables['intra'] = table_dict['CIF_intra']
        bits_tables['inter'] = table_dict['CIF_inter']
    elif w == 176 and h == 144:
        bits_tables['intra'] = table_dict['QCIF_intra']
        bits_tables['inter'] = table_dict['QCIF_inter']
    if RCFlag == 0:
        A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                                  FMEEnable, FastME, RCFlag, frame)
    elif RCFlag == 1:
        encode_RC_1(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable, FastME,
                    RCFlag, targetBR, fps, bits_tables, frame)


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
                qp = search_qp(budget_per_row, bits_tables['intra'])
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
    # first run
    q = quantization.generate_q(block_size, qp)
    if not VBSEnable:
        for x in range(num_frames):
            print('encode frame: ' + str(x))
            bit_sum = 0
            if x % period == 0:
                res, prediction, vec, quan_frame = prediction_encode.intra_residual(frame_block_array[x], n, q)
                code, bit_count = entropy_encode.entropy_encode_quan_frame_block(quan_frame)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                bit_sum += bit_count
                prediction = blocking.deblock_frame(prediction, w, h)
                prediction_array = [prediction]
            else:
                res, vec = prediction_encode.generate_residual_ME(prediction_array, frame_block_array[x], w, h, n,
                                                                  r, FMEEnable, FastME)
                tran = transform_encode.transform_frame(res)
                quan = quantization_encode.quantization_frame(tran, q)
                code, bit_count = entropy_encode.entropy_encode_quan_frame_block(quan)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                bit_sum += bit_count
                # decode start
                dequan = quantization_decode.dequantization_frame(quan, q)
                itran = transform_decode.inverse_transform_frame(dequan).clip(-128, 127)
                res = blocking.deblock_frame(itran, w, h)
                prediction = prediction_decode.decode_residual_ME(prediction_array, res, vec, w, h, block_size, FMEEnable)
                prediction_array.insert(0, prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
            bit_count_arr.append(bit_sum)
    else:
        lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
        if qp > 0:
            q_split = quantization.generate_q(int(block_size / 2), qp - 1)
        else:
            q_split = quantization.generate_q(int(block_size / 2), qp)
        for x in range(num_frames):
            print('encode frame: ' + str(x))
            bit_sum = 0
            # here the transform and quantization are done within the prediction part
            # as it needs these information to decide if sub blocks takes less r-d cost
            if x % period == 0:
                prediction, vec, split, res_code = prediction_encode.intra_residual_VBS(frame_block_array[x], n,
                                                                                        lambda_val, q, q_split)
                residual_file.write(res_code)
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                diff_file.write(code)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_sum += bit_count
                prediction = blocking.deblock_frame(prediction, w, h)
                prediction_array = [prediction]
            else:
                block_itran, vec, split, res_code = prediction_encode.generate_residual_ME_VBS(prediction_array,
                                                                                               frame_block_array[x], w,
                                                                                               h, n, r, lambda_val, q,
                                                                                               q_split, FMEEnable,
                                                                                               FastME)
                residual_file.write(res_code)
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                diff_file.write(code)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_sum += bit_count
                res = blocking.deblock_frame(block_itran)
                prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec, split, w, h, block_size,
                                                                      FMEEnable)
                prediction_array.insert(0, prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
            bit_count_arr.append(bit_sum)

    residual_file.close()
    diff_file.close()

def decode_complete(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
    w, h, i, qp, period, VBSEnable, FMEEnable, RCFlag = entropy_decode.decode_setting(setting)
    if RCFlag == 0:
        A2process.decode_complete(filepath)
        return
    elif RCFlag == 1:
        decode_RC(filepath)


def decode_RC(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, block_size, qp, period, VBSEnable, FMEEnable, RCFlag = entropy_decode.decode_setting(setting)
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
