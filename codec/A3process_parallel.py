from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from codec import blocking, evaluation, quantization
from codec.decoder import entropy_decode, prediction_decode, differential_decode, quantization_decode, transform_decode
from codec.encoder import entropy_encode, prediction_encode_parallel, prediction_encode_row, differential_encode
from utils import reader
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

import time

def encode_parallel_mode_1(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable,
                           VBSEnable, encode_executor):
    total_time = 0
    block_size = len(frame_block[0][0])
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    task_handles = []
    diff_file_code = ''
    res_code_append = ''
    block_itran = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
    
    with SharedMemoryManager() as smm:
        np_array = prediction_array[0]
        pa_shape, pa_dtype = np_array.shape, np_array.dtype
        pa_shared = smm.SharedMemory(size=np_array.nbytes)
        shm_pa_np = np.ndarray(pa_shape, dtype=pa_dtype, buffer=pa_shared.buf)
        shm_pa_np[:] = np_array[:]

        fb_shared = smm.SharedMemory(size=frame_block.nbytes)
        fb_shape, fb_dtype = frame_block.shape, frame_block.dtype
        shm_fb_np = np.ndarray(fb_shape, dtype=fb_dtype, buffer=fb_shared.buf)
        shm_fb_np[:] = frame_block[:]
        for i in range(n_rows_frame):
            for j in range(n_cols_frame):
                # task = encode_executor.submit(prediction_encode_parallel.generate_residual_ME_block,
                #                               prediction_array, block_size, frame_block, w, h, n, r, lambda_val,
                #                               q_non_split, q_split, FMEEnable,
                #                               VBSEnable, i, j)
                task = encode_executor.submit(prediction_encode_parallel.generate_residual_ME_block_shared_memory,
                                              block_size, w, h, n, r, lambda_val,
                                              q_non_split.copy(), q_split.copy(), FMEEnable,
                                              VBSEnable, i, j, pa_shared.name, pa_shape, pa_dtype, fb_shared.name, fb_shape, fb_dtype)
                task_handles.append(task)


    for i in range(n_rows_frame):
        for j in range(n_cols_frame):
            # print('get block' + str(i) + ',' + str(j))
            index = i * n_cols_frame + j
            itran_single_block, vec_code, split_code, res_code = task_handles[index].result()
            block_itran[i][j] = itran_single_block
            if VBSEnable:
                diff_file_code += split_code
            diff_file_code += vec_code
            res_code_append += res_code
    split_array = []
    vec_array = np.empty((0, 3), dtype=np.uint16)
    while len(diff_file_code) > 0:
        if VBSEnable:
            split, diff_file_code = entropy_decode.get_num(diff_file_code, 1)
            split_array.append(split[0])
        else:
            split = [0]
            split_array.append(0)
        vec, diff_file_code = entropy_decode.decode_vec_one_frame_alter(diff_file_code,
                                                                        len(split) + 3 * np.sum(split), True)
        vec_array = np.append(vec_array, vec, axis=0)
    res = blocking.deblock_frame(block_itran)
    prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array, split_array, w, h,
                                                          block_size,
                                                          FMEEnable)

    return prediction, diff_file_code, res_code_append


def encode_intra_mode_2(frame_block, w, h, n, lambda_val, q_non_split, q_split,
                          VBSEnable, encode_executor):
    block_size = len(frame_block[0][0])
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    diff_file_code = ''
    res_code_append = ''
    prediction = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)
    res_code_buffer = [[None] * n_cols_frame for k in range(n_rows_frame)]
    mode_buffer = [[None] * n_cols_frame for k in range(n_rows_frame)]
    split_buffer = [[None] * n_cols_frame for k in range(n_rows_frame)]
    for mysum in range(n_rows_frame + n_cols_frame - 1):
        task_handles = []
        pairs = []
        for i in range(n_rows_frame):
            j = mysum - i
            if j < 0 or j >= n_cols_frame:
                continue
            pairs.append([i, j])
            if j == 0:
                prev_split = 0
                prev_mode = 0
            else:
                prev_split = split_buffer[i][j - 1]
                prev_mode = mode_buffer[i][j - 1][-1]
            task = encode_executor.submit(prediction_encode_parallel.intra_residual_block_parallel, frame_block, n,
                                          lambda_val, q_non_split.copy(), q_split.copy(), VBSEnable, prediction, i, j, prev_mode,
                                          prev_split)
            task_handles.append(task)
        for x in range(len(pairs)):
            i = pairs[x][0]
            j = pairs[x][1]
            task = task_handles[x]
            pred_block, mode_array, split_array, entropy_encode_str = task.result()
            prediction[i][j] = pred_block[i][j]
            mode_buffer[i][j] = mode_array
            split_buffer[i][j] = split_array[0]
            res_code_buffer[i][j] = entropy_encode_str
    for i in range(n_rows_frame):
        mode_row = []
        split_row = []
        for j in range(n_cols_frame):
            res_code_append += res_code_buffer[i][j]
            mode_row += mode_buffer[i][j]
            split_row.append(split_buffer[i][j])
        if VBSEnable:
            code, bit_count = entropy_encode.entropy_encode_vec_alter(
                differential_encode.differential_encode(split_row))
            diff_file_code += code
        code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(mode_row))
        diff_file_code += code
    prediction = blocking.deblock_frame(prediction, w, h)
    return prediction, diff_file_code, res_code_append


def encode_inter_mode_2(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable, FastME,
                        VBSEnable, encode_executor):
    block_size = len(frame_block[0][0])
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    task_handles = []
    diff_file_code = ''
    res_code_append = ''
    vec_array = []
    split_array = []
    block_itran = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
    for i in range(n_rows_frame):
        task = encode_executor.submit(prediction_encode_parallel.inter_residual_row_parallel, prediction_array,
                                      frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable,
                                      FastME, VBSEnable, block_itran, i)
        task_handles.append(task)
    for i in range(n_rows_frame):
        itran_row, diff_code, res_code, split, vector = task_handles[i].result()
        block_itran[i] = itran_row
        diff_file_code += diff_code
        res_code_append += res_code
        vec_array += vector
        split_array += split
    res = blocking.deblock_frame(block_itran)
    prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array, split_array, w, h,
                                                          block_size,
                                                          FMEEnable)
    return prediction, diff_file_code, res_code_append


# parallel within frame
def encode_parallel_1_2(filepath, w, h, block_size, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                        FMEEnable, FastME, ParallelMode, num_frames=None):
    encode_executor = ProcessPoolExecutor(max_workers=16)
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, block_size, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    # config is written to the first line of diff file
    line, bits = entropy_encode.entropy_encode_setting(w, h, block_size, qp, period, VBSEnable, FMEEnable,
                                                       ParallelMode=ParallelMode)
    diff_file.write(line)
    diff_file.write("\n")
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction_array = []
    prediction_to_file = []
    if ParallelMode == 1:
        prediction_array.append(np.full((h, w), 128, dtype=np.uint8))
    lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
    q = quantization.generate_q(block_size, qp)
    q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        if ParallelMode == 1:
            prediction, diff_file_code, res_file_code = encode_parallel_mode_1(prediction_array,
                                                                               frame_block_array[x], w, h, n, r,
                                                                               lambda_val, q, q_split,
                                                                               FMEEnable, VBSEnable, encode_executor)
            diff_file.write(diff_file_code)
            residual_file.write(res_file_code)
            prediction_array.insert(0, prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        elif x % period == 0:
            prediction, diff_file_code, res_file_code = encode_intra_mode_2(frame_block_array[x], w, h, n,
                                                                            lambda_val,
                                                                            q, q_split,
                                                                            VBSEnable, encode_executor)
            diff_file.write(diff_file_code)
            residual_file.write(res_file_code)
            prediction_array = [prediction]
        else:
            prediction, diff_file_code, res_file_code = encode_inter_mode_2(prediction_array, frame_block_array[x], w,
                                                                            h, n, r, lambda_val,
                                                                            q, q_split, FMEEnable, FastME,
                                                                            VBSEnable, encode_executor)
            diff_file.write(diff_file_code)
            residual_file.write(res_file_code)
            prediction_array.insert(0, prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        prediction_to_file.append(prediction)
    encode_executor.shutdown()
    residual_file.close()
    diff_file.close()
    return prediction_to_file, None


# parallel over frames
def encode_parallel_3(filepath, w, h, block_size, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient,
                      FMEEnable, FastME, ParallelMode, num_frames=None):
    FastME = False
    encode_executor = ProcessPoolExecutor(max_workers=16)
    th_ex = ThreadPoolExecutor(max_workers=16)
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, block_size, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    # config is written to the first line of diff file
    line, bits = entropy_encode.entropy_encode_setting(w, h, block_size, qp, period, VBSEnable, FMEEnable,
                                                       ParallelMode=ParallelMode)
    diff_file.write(line)
    diff_file.write("\n")
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction_array = []
    prediction_to_file = []
    n_rows_frame = (h - 1) // block_size + 1
    n_cols_frame = (w - 1) // block_size + 1
    lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
    q = quantization.generate_q(block_size, qp)
    q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
    task_handles = [[None] * n_rows_frame for k in range(2)]
    for x in range(((num_frames - 1) // 2) + 1):
        x_1 = x * 2
        x_2 = x * 2 + 1
        print('encode frame pair: ' + str(x_1) + ' and ' + str(x_2))
        block_itran_1 = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
        block_itran_2 = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
        if x_1 % period == 0 or (x_2 % period == 0 and x_2 < num_frames):
            prediction_intra = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)
        prediction_array_temp = [np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.uint8)] + prediction_array
        if len(prediction_array_temp) >= nRefFrames:
            prediction_array_temp = prediction_array_temp[:nRefFrames]
        frame_2_diff = ''
        frame_2_res = ''
        vec_array_1 = []
        vec_array_2 = []
        split_array_1 = []
        split_array_2 = []
        for i_1 in range(n_rows_frame + 3):
            print(i_1)
            i_2 = i_1 - 3
            if i_1 < n_rows_frame:
                if x_1 % period == 0:
                    task = th_ex.submit(prediction_encode_row.intra_residual_row, frame_block_array[x], n,
                                                                                            lambda_val, q, q_split,
                                                                                            VBSEnable, prediction_intra, i_1)
                else:
                    task = th_ex.submit(prediction_encode_row.generate_residual_ME_row, prediction_array,
                                                                                                   frame_block_array[x],
                                                                                                   w,
                                                                                                   h, n, r, lambda_val,
                                                                                                   q, q_split,
                                                                                                   FMEEnable,
                                                                                                   FastME, VBSEnable,
                                                                                                   block_itran_1, i_1)
                task_handles[0][i_1] = task
            if i_2 >= 0 and x_2 < num_frames:
                if x_2 % period == 0:
                    task = th_ex.submit(prediction_encode_row.intra_residual_row, frame_block_array[x], n,
                                                  lambda_val, q, q_split,
                                                  VBSEnable, prediction_intra, i_2)
                else:
                    task = th_ex.submit(prediction_encode_row.generate_residual_ME_row, prediction_array_temp,
                                                  frame_block_array[x],
                                                  w,
                                                  h, n, r, lambda_val,
                                                  q, q_split,
                                                  FMEEnable,
                                                  FastME, VBSEnable,
                                                  block_itran_2, i_2)
                task_handles[1][i_2] = task
            # receive
            if i_1 < n_rows_frame:
                if x_1 % period == 0:
                    prediction_intra, vec, split, res_code = task_handles[0][i_1].result()
                    prediction_array_temp[0] = blocking.deblock_frame(prediction_intra, w, h)
                else:
                    block_itran_1, vec, split, res_code = task_handles[0][i_1].result()
                    prediction_array_temp[0] = blocking.deblock_frame(block_itran_1, w, h)
                    vec_array_1 += vec
                    split_array_1 += split
                residual_file.write(res_code)
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    diff_file.write(code)
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                diff_file.write(code)
            if i_2 >= 0 and x_2 < num_frames:
                if x_2 % period == 0:
                    prediction_intra, vec, split, res_code = task_handles[1][i_2].result()
                else:
                    block_itran_2, vec, split, res_code = task_handles[1][i_2].result()
                    vec_array_2 += vec
                    split_array_2 += split
                frame_2_res += res_code
                if VBSEnable:
                    code, bit_count = entropy_encode.entropy_encode_vec_alter(
                        differential_encode.differential_encode(split))
                    frame_2_diff += code
                code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vec))
                frame_2_diff += code
        if x_2 < num_frames:
            diff_file.write(frame_2_diff)
            residual_file.write(frame_2_res)
        if x_1 % period == 0:
            prediction_array = [blocking.deblock_frame(prediction_intra, w, h)]
        else:
            res = blocking.deblock_frame(block_itran_1)
            prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array_1, split_array_1, w, h,
                                                                  block_size,
                                                                  FMEEnable)
            prediction_array.insert(0, prediction)
            prediction_to_file.append(prediction)
            if len(prediction_array) >= nRefFrames:
                prediction_array = prediction_array[:nRefFrames]
        if x_2 < num_frames:
            if x_2 % period == 0:
                prediction_array = [blocking.deblock_frame(prediction_intra, w, h)]
            else:
                res = blocking.deblock_frame(block_itran_2)
                prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec_array_2,
                                                                                    split_array_2, w, h,
                                                                                    block_size,
                                                                                    FMEEnable)
                prediction_array.insert(0, prediction)
                prediction_to_file.append(prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
        # prediction_to_file.append(prediction)
    encode_executor.shutdown()
    residual_file.close()
    diff_file.close()
    return prediction_to_file, None


def decode_parallel_1(filepath):
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
    recon_array.append(np.full((h, w), 128, dtype=np.uint8))
    n_rows_frame = (h - 1) // block_size + 1  # n_block_h
    n_cols_frame = (w - 1) // block_size + 1  # n_block_w
    frame_count = 0
    q = quantization.generate_q(block_size, qp)
    q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
    while len(res_code) > 0 and len(vec_code) > 0:
        print("decode frame: " + str(frame_count))
        quan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size))
        dequan_frame = np.zeros((n_rows_frame, n_cols_frame, block_size, block_size), dtype=np.int16)
        vec_array = np.empty((0, 3), dtype=np.uint16)  # vector or mode
        split_array = np.empty(0, dtype=np.uint16)
        for i in range(n_rows_frame):
            for j in range(n_cols_frame):
                if VBSEnable:
                    # get the row of split indicators
                    split_block, vec_code = entropy_decode.get_num(vec_code, 1)
                else:
                    split_block = np.zeros(1)
                array_len = len(split_block) + 3 * np.sum(split_block)

                quan_frame, res_code = entropy_decode.decode_quan_frame_VBS_given_block(res_code, block_size,
                                                                                        split_block, quan_frame, i, j)
                dequan_frame = quantization_decode.dequantization_frame_VBS_given_block(quan_frame, q, q_split,
                                                                                        split_block,
                                                                                        dequan_frame, i, j)

                # extract mode array or vector array
                vec_block, vec_code = entropy_decode.decode_vec_one_frame_alter(vec_code, array_len, True)
                vec_array = np.append(vec_array, np.array(vec_block), axis=0)
                split_array = np.append(split_array, split_block)
        itran = transform_decode.inverse_transform_frame_VBS(dequan_frame, split_array)
        res = blocking.deblock_frame(itran, w, h)
        recon = prediction_decode.decode_residual_ME_VBS(recon_array, res, vec_array, split_array, w, h, block_size,
                                                         FMEEnable)
        recon_array.insert(0, recon)
        # here use the upper limit of nRefFrame for easier coding purpose
        if len(recon_array) == 4:
            recon_array = recon_array[:4]
        video.append(recon)
        frame_count += 1
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')


def decode_parallel_2(filepath):
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
    recon_array.append(np.full((h, w), 128, dtype=np.uint8))
    n_rows_frame = (h - 1) // block_size + 1  # n_block_h
    n_cols_frame = (w - 1) // block_size + 1  # n_block_w
    frame_count = 0
    q = quantization.generate_q(block_size, qp)
    q_split = quantization.generate_q(int(block_size / 2), max(qp - 1, 0))
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
