import numpy as np

from codec import evaluation
from codec.decoder import quantization_decode, transform_decode
from codec.encoder import transform_encode, quantization_encode, entropy_encode, prediction_encode_row, \
    differential_encode
from codec.encoder.prediction_encode import search_motion_non_fraction, search_motion_fraction, closest_multi_power2


def generate_residual_ME_block(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable,
                               VBSEnable, i, j):
    block_size = len(frame_block[0][0])
    vector_array = []
    half_block_size = int(block_size / 2)
    code_str_entropy = ''
    i_h = i * block_size
    i_w = j * block_size
    if not FMEEnable:
        min_MAE, block_origin, vec_non_split = search_motion_non_fraction(w, h, i_h, i_w, block_size, r,
                                                                          prediction_array,
                                                                          frame_block[i][j], False)
    else:
        min_MAE, block_origin, vec_non_split = search_motion_fraction(w, h, i_h, i_w, block_size, r,
                                                                      prediction_array,
                                                                      frame_block[i][j], False)
    # residual bits and ssd
    block = np.zeros((block_size, block_size), dtype=np.int16)
    for x in range(block_size):
        for y in range(block_size):
            block[x][y] = closest_multi_power2(block_origin[x][y], n)
    tran = transform_encode.transform_block(block)
    quan = quantization_encode.quantization_block(tran, q_non_split)
    code_str_non_split, bits_non_split = entropy_encode.entropy_encode_single_block(quan.astype(np.int16))
    dequan = quantization_decode.dequantization_block(quan, q_non_split)
    itran_non_split = transform_decode.inverse_transform_block(dequan)
    ssd = evaluation.calculate_ssd(itran_non_split, block_origin)
    # vector part
    code, bits_vec = entropy_encode.entropy_encode_single_vec(vec_non_split)
    bits_non_split += bits_vec
    # mode indicate part
    code, bits_mode = entropy_encode.exp_golomb(0)
    bits_non_split += bits_mode
    r_d_score_non_split = evaluation.calculate_rdo(ssd, lambda_val, bits_non_split)
    if VBSEnable:
        # split
        vec_arr_split = []
        code_str_split = ''
        bits_split_sum = 0
        ssd_split_sum = 0
        block = np.zeros((half_block_size, half_block_size), dtype=np.int16)
        itran_split = np.zeros((block_size, block_size), dtype=np.int16)
        # mode only needed once for split 4 blocks
        code, bits_mode = entropy_encode.exp_golomb(1)
        bits_split_sum += bits_mode
        for slice_x in [0, half_block_size]:
            for slice_y in [0, half_block_size]:
                if not FMEEnable:
                    min_MAE, block_origin, vec = search_motion_non_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                            prediction_array,
                                                                            frame_block[i][j][
                                                                            slice_x:slice_x + half_block_size,
                                                                            slice_y:slice_y + half_block_size],
                                                                            False)
                else:
                    min_MAE, block_origin, vec = search_motion_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                        prediction_array,
                                                                        frame_block[i][j][
                                                                        slice_x:slice_x + half_block_size,
                                                                        slice_y:slice_y + half_block_size], False)
                vec_arr_split.append(vec)
                for x in range(half_block_size):
                    for y in range(half_block_size):
                        block[x][y] = closest_multi_power2(block_origin[x][y], n)
                tran = transform_encode.transform_block(block)
                quan = quantization_encode.quantization_block(tran, q_split)
                code_str, bits = entropy_encode.entropy_encode_single_block(quan.astype(np.int16))
                dequan = quantization_decode.dequantization_block(quan, q_split)
                itran = transform_decode.inverse_transform_block(dequan)
                # ssd is cumulative across the sub blocks
                ssd = evaluation.calculate_ssd(itran, block_origin)
                ssd_split_sum += ssd
                # vector encode est
                code, bits_vec = entropy_encode.entropy_encode_single_vec(vec)
                bits += bits_vec
                bits_split_sum += bits
                code_str_split += code_str
                itran_split[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size] = itran
            r_d_score_split = evaluation.calculate_rdo(ssd_split_sum, lambda_val, bits_split_sum)
    else:
        r_d_score_split = r_d_score_non_split
    # the estimated score is proportional to ssd and number of bits,
    # which is smaller for better
    if r_d_score_non_split <= r_d_score_split:
        split_indicator = 0
        vector_array.append(vec_non_split)
        code_str_entropy += code_str_non_split
        block_itran = itran_non_split
    else:
        split_indicator = 1
        vector_array += vec_arr_split
        code_str_entropy += code_str_split
        block_itran = itran_split

    return block_itran, entropy_encode.entropy_encode_vec_alter(vector_array)[0], entropy_encode.exp_golomb(
        split_indicator)[0], code_str_entropy


def intra_residual_block_parallel(frame_block, n, lambda_val, q_non_split, q_split, VBSEnable, pred, i, j, prev_mode, prev_split):
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    # 0 for horizontal, 1 for vertical
    mode_array = []
    split_array = []
    entropy_encode_str = ''
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    blank_half = np.full((half_block_size, half_block_size), 128, dtype=np.int16)
    # non split
    curr_block = frame_block[i][j]
    # vertical
    prediction_block_ver = blank
    if i != 0:
        prev = pred[i - 1][j]
        prediction_block_ver = np.tile(prev[block_size - 1], (block_size, 1))
    diff_ver = np.subtract(curr_block.astype(np.int16), prediction_block_ver.astype(np.int16))
    MAE_ver = np.sum(np.abs(diff_ver))
    # horizontal
    prediction_block_hor = blank
    if j != 0:
        prev = pred[i][j - 1]
        prediction_block_hor = np.tile(prev[:, block_size - 1], (block_size, 1)).transpose()
    diff_hor = np.subtract(curr_block.astype(np.int16), prediction_block_hor.astype(np.int16))
    MAE_hor = np.sum(np.abs(diff_hor))
    if MAE_ver < MAE_hor:
        mode_non_split = 1
        res_original = diff_ver
        prediction_block_non_split = prediction_block_ver
    else:
        mode_non_split = 0
        res_original = diff_hor
        prediction_block_non_split = prediction_block_hor
    # residual bits and ssd
    block = np.zeros((block_size, block_size), dtype=np.int16)
    for x in range(block_size):
        for y in range(block_size):
            block[x][y] = closest_multi_power2(res_original[x][y], n)
    tran = transform_encode.transform_block(block)
    quan_non_split = quantization_encode.quantization_block(tran, q_non_split)
    code_str_non_split, bits_non_split = entropy_encode.entropy_encode_single_block(quan_non_split.astype(np.int16))
    dequan = quantization_decode.dequantization_block(quan_non_split, q_non_split)
    itran_non_split = transform_decode.inverse_transform_block(dequan)
    ssd = evaluation.calculate_ssd(itran_non_split, res_original)
    # mode part
    code, bits_mode = entropy_encode.exp_golomb(mode_non_split - prev_mode)
    bits_non_split += bits_mode
    # split indicator part
    code, bits_indicator = entropy_encode.exp_golomb(0 - prev_split)
    bits_non_split += bits_indicator
    r_d_score_non_split = evaluation.calculate_rdo(ssd, lambda_val, bits_non_split)
    if VBSEnable:
        # split
        code_str_split = ''
        bits_split_sum = 0
        ssd_split_sum = 0
        prev_mode_split = prev_mode
        mode_array_split = []
        prediction_block_split = np.zeros((block_size, block_size), dtype=np.uint8)
        block = np.zeros((half_block_size, half_block_size), dtype=np.int16)
        # split indicator bits
        code, bits_mode = entropy_encode.exp_golomb(1 - prev_split)
        bits_split_sum += bits_mode
        for slice_x in [0, half_block_size]:
            for slice_y in [0, half_block_size]:
                curr_half_block = curr_block[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size]
                if slice_x == 0:
                    if i == 0:
                        prediction_block_ver = blank_half
                    else:
                        prev = pred[i - 1][j]
                        prediction_block_ver = np.tile(prev[block_size - 1, slice_y:slice_y + half_block_size],
                                                       (half_block_size, 1))
                else:
                    prediction_block_ver = np.tile(
                        prediction_block_split[slice_x - 1, slice_y:slice_y + half_block_size],
                        (half_block_size, 1))
                diff_ver = np.subtract(curr_half_block.astype(np.int16), prediction_block_ver.astype(np.int16))
                MAE_ver = np.sum(np.abs(diff_ver))
                if slice_y == 0:
                    if j == 0:
                        prediction_block_hor = blank_half
                    else:
                        prev = pred[i][j - 1]
                        prediction_block_hor = np.tile(prev[slice_x:slice_x + half_block_size, block_size - 1],
                                                       (half_block_size, 1)).transpose()
                else:
                    prediction_block_hor = np.tile(
                        prediction_block_split[slice_x:slice_x + half_block_size, slice_y - 1],
                        (half_block_size, 1)).transpose()
                diff_hor = np.subtract(curr_half_block.astype(np.int16), prediction_block_hor.astype(np.int16))
                MAE_hor = np.sum(np.abs(diff_hor))
                if MAE_ver < MAE_hor:
                    mode_split = 1
                    res_original = diff_ver
                    prediction_sub_block = prediction_block_ver
                else:
                    mode_split = 0
                    res_original = diff_hor
                    prediction_sub_block = prediction_block_hor
                for x in range(half_block_size):
                    for y in range(half_block_size):
                        block[x][y] = closest_multi_power2(res_original[x][y], n)
                tran = transform_encode.transform_block(block)
                quan = quantization_encode.quantization_block(tran, q_split)
                code_str, bits_split = entropy_encode.entropy_encode_single_block(quan.astype(np.int16))
                code_str_split += code_str
                bits_split_sum += bits_split
                dequan = quantization_decode.dequantization_block(quan, q_split)
                itran_split = transform_decode.inverse_transform_block(dequan)
                ssd = evaluation.calculate_ssd(itran_split, res_original)
                ssd_split_sum += ssd
                # mode part
                code, bits_mode = entropy_encode.exp_golomb(mode_split - prev_mode_split)
                bits_non_split += bits_mode
                # ready for next iteration
                mode_array_split.append(mode_split)
                prev_mode_split = mode_split
                prediction_block_split[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size] = \
                    np.add(prediction_sub_block, itran_split).clip(0, 255).astype(np.uint8)
            r_d_score_split = evaluation.calculate_rdo(ssd_split_sum, lambda_val, bits_split_sum)
    else:
        r_d_score_split = r_d_score_non_split
    # decide
    if r_d_score_non_split <= r_d_score_split:
        split_array.append(0)
        mode_array.append(mode_non_split)
        pred[i][j] = np.add(prediction_block_non_split, itran_non_split).clip(0, 255).astype(np.uint8)
        entropy_encode_str += code_str_non_split
    else:
        split_array.append(1)
        mode_array += mode_array_split
        pred[i][j] = prediction_block_split
        entropy_encode_str += code_str_split
    # do encoding and bit counting

    return pred, mode_array, split_array, entropy_encode_str

def intra_residual_row_parallel(frame_block, n, lambda_val, q_non_split, q_split, VBSEnable, prediction, i):
    prediction_row, mode_array, split_array, res_code = prediction_encode_row.intra_residual_row(frame_block, n,
                                                                                                 lambda_val,
                                                                                                 q_non_split, q_split,
                                                                                                 VBSEnable, prediction,
                                                                                                 i)
    diff_file_code = ''
    if VBSEnable:
        code, bit_count = entropy_encode.entropy_encode_vec_alter(
            differential_encode.differential_encode(split_array))
        diff_file_code += code
    code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(mode_array))
    diff_file_code += code
    return prediction_row[i], diff_file_code, res_code


def inter_residual_row_parallel(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable,
                                FastME, VBSEnable, block_itran, i):
    itran_row, vector_array, split_array, res_code = prediction_encode_row.generate_residual_ME_row(prediction_array,
                                                                                                    frame_block,
                                                                                                    w,
                                                                                                    h, n, r, lambda_val,
                                                                                                    q_non_split,
                                                                                                    q_split,
                                                                                                    FMEEnable,
                                                                                                    FastME, VBSEnable,
                                                                                                    block_itran, i)
    diff_file_code = ''
    if VBSEnable:
        code, bit_count = entropy_encode.entropy_encode_vec_alter(
            differential_encode.differential_encode(split_array))
        diff_file_code += code
    code, bit_count = entropy_encode.entropy_encode_vec_alter(differential_encode.differential_encode(vector_array))
    diff_file_code += code
    return itran_row[i], diff_file_code, res_code, split_array, vector_array
