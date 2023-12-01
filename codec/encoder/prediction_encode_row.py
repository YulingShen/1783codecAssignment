import numpy as np

from codec import evaluation
from codec.decoder import transform_decode, quantization_decode
from codec.encoder import entropy_encode, transform_encode, quantization_encode
from codec.encoder.prediction_encode import search_motion_non_fraction, search_motion_fraction, closest_multi_power2


def generate_residual_ME_row(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable,
                             FastME, VBSEnable, block_itran, i):
    block_size = len(frame_block[0][0])
    n_w = len(frame_block[0])
    vector_array = []
    split_array = []
    half_block_size = int(block_size / 2)
    code_str_entropy = ''
    prev_vec = [0, 0, 0]
    prev_split = 0
    mvp = [0, 0]
    for j in range(n_w):
        # top left of the block
        # non split
        i_h = i * block_size
        i_w = j * block_size
        if not FMEEnable:
            min_MAE, block_origin, vec_non_split = search_motion_non_fraction(w, h, i_h, i_w, block_size, r,
                                                                              prediction_array,
                                                                              frame_block[i][j], FastME, mvp)
        else:
            min_MAE, block_origin, vec_non_split = search_motion_fraction(w, h, i_h, i_w, block_size, r,
                                                                          prediction_array,
                                                                          frame_block[i][j], FastME, mvp)
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
        code, bits_vec = entropy_encode.entropy_encode_single_vec(np.subtract(vec_non_split, prev_vec))
        bits_non_split += bits_vec
        # mode indicate part
        code, bits_mode = entropy_encode.exp_golomb(0 - prev_split)
        bits_non_split += bits_mode
        r_d_score_non_split = evaluation.calculate_rdo(ssd, lambda_val, bits_non_split)
        if VBSEnable:
            # split
            vec_arr_split = []
            code_str_split = ''
            bits_split_sum = 0
            ssd_split_sum = 0
            prev_split_vec = prev_vec
            mvp_split = mvp
            block = np.zeros((half_block_size, half_block_size), dtype=np.int16)
            itran_split = np.zeros((block_size, block_size), dtype=np.int16)
            # mode only needed once for split 4 blocks
            code, bits_mode = entropy_encode.exp_golomb(1 - prev_split)
            bits_split_sum += bits_mode
            for slice_x in [0, half_block_size]:
                for slice_y in [0, half_block_size]:
                    if not FMEEnable:
                        min_MAE, block_origin, vec = search_motion_non_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                                prediction_array,
                                                                                frame_block[i][j][
                                                                                slice_x:slice_x + half_block_size,
                                                                                slice_y:slice_y + half_block_size],
                                                                                FastME, mvp_split)
                    else:
                        min_MAE, block_origin, vec = search_motion_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                            prediction_array,
                                                                            frame_block[i][j][
                                                                            slice_x:slice_x + half_block_size,
                                                                            slice_y:slice_y + half_block_size], FastME,
                                                                            mvp_split)
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
                    code, bits_vec = entropy_encode.entropy_encode_single_vec(np.subtract(vec, prev_split_vec))
                    bits += bits_vec
                    bits_split_sum += bits
                    code_str_split += code_str
                    prev_split_vec = bits_vec
                    itran_split[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size] = itran
                    mvp_split = vec[:2]
                r_d_score_split = evaluation.calculate_rdo(ssd_split_sum, lambda_val, bits_split_sum)
        else:
            r_d_score_split = r_d_score_non_split
        # the estimated score is proportional to ssd and number of bits,
        # which is smaller for better
        if r_d_score_non_split <= r_d_score_split:
            split_array.append(0)
            vector_array.append(vec_non_split)
            code_str_entropy += code_str_non_split
            prev_vec = vec_non_split
            prev_split = 0
            block_itran[i][j] = itran_non_split
            mvp = vec_non_split[:2]
        else:
            split_array.append(1)
            vector_array += vec_arr_split
            code_str_entropy += code_str_split
            prev_vec = prev_split_vec
            prev_split = 1
            block_itran[i][j] = itran_split
            mvp = mvp_split
    return block_itran, vector_array, split_array, code_str_entropy


def generate_residual_ME_row_leverage(prediction_array, frame_block, w, h, n, r, q_non_split, q_split, FMEEnable,
                                      block_itran, i, split_array, vector_reference):
    block_size = len(frame_block[0][0])
    n_w = len(frame_block[0])
    vector_array = []
    half_block_size = int(block_size / 2)
    code_str_entropy = ''
    vector_counter = 0
    for j in range(n_w):
        split_mode = split_array[j]
        if split_mode == 0:
            # non split
            i_h = i * block_size
            i_w = j * block_size
            if not FMEEnable:
                min_MAE, block_origin, vec_non_split = search_motion_non_fraction(w, h, i_h, i_w, block_size, r,
                                                                                  prediction_array,
                                                                                  frame_block[i][j], True,
                                                                                  vector_reference[vector_counter])
            else:
                min_MAE, block_origin, vec_non_split = search_motion_fraction(w, h, i_h, i_w, block_size, r,
                                                                              prediction_array,
                                                                              frame_block[i][j], True,
                                                                              vector_reference[vector_counter])
            vector_counter += 1
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

            vector_array.append(vec_non_split)
            code_str_entropy += code_str_non_split
            block_itran[i][j] = itran_non_split
        else:
            # split
            vec_arr_split = []
            code_str_split = ''
            block = np.zeros((half_block_size, half_block_size), dtype=np.int16)
            itran_split = np.zeros((block_size, block_size), dtype=np.int16)
            # mode only needed once for split 4 blocks
            for slice_x in [0, half_block_size]:
                for slice_y in [0, half_block_size]:
                    if not FMEEnable:
                        min_MAE, block_origin, vec = search_motion_non_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                                prediction_array,
                                                                                frame_block[i][j][
                                                                                slice_x:slice_x + half_block_size,
                                                                                slice_y:slice_y + half_block_size],
                                                                                True, vector_reference[vector_counter])
                    else:
                        min_MAE, block_origin, vec = search_motion_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                            prediction_array,
                                                                            frame_block[i][j][
                                                                            slice_x:slice_x + half_block_size,
                                                                            slice_y:slice_y + half_block_size], True,
                                                                            vector_reference[vector_counter])
                    vector_counter += 1
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
                    # vector encode est
                    code_str_split += code_str
                    itran_split[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size] = itran

                    vector_array += vec_arr_split
                    code_str_entropy += code_str_split
                    block_itran[i][j] = itran_split
        # the estimated score is proportional to ssd and number of bits,
        # which is smaller for better
    return block_itran, vector_array, split_array, code_str_entropy


def intra_residual_row(frame_block, n, lambda_val, q_non_split, q_split, VBSEnable, pred, i):
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    n_w = len(frame_block[0])
    # 0 for horizontal, 1 for vertical
    mode_array = []
    split_array = []
    prev_mode = 0
    prev_split = 0
    entropy_encode_str = ''
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    blank_half = np.full((half_block_size, half_block_size), 128, dtype=np.int16)
    for j in range(n_w):
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
                    prev_mode_split = mode_split
                    mode_array_split.append(mode_split)
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
            prev_split = 0
        else:
            split_array.append(1)
            mode_array += mode_array_split
            pred[i][j] = prediction_block_split
            entropy_encode_str += code_str_split
            prev_split = 1
    # do encoding and bit counting

    return pred, mode_array, split_array, entropy_encode_str


def intra_residual_row_leverage(frame_block, n, lambda_val, q_non_split, q_split, VBSEnable, pred, i, split_array):
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    n_w = len(frame_block[0])
    # 0 for horizontal, 1 for vertical
    mode_array = []
    entropy_encode_str = ''
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    blank_half = np.full((half_block_size, half_block_size), 128, dtype=np.int16)
    for j in range(n_w):
        split_mode = split_array[j]
        if split_mode == 0:
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

            mode_array.append(mode_non_split)
            pred[i][j] = np.add(prediction_block_non_split, itran_non_split).clip(0, 255).astype(np.uint8)
            entropy_encode_str += code_str_non_split
        else:
            # split
            code_str_split = ''
            bits_split_sum = 0
            mode_array_split = []
            prediction_block_split = np.zeros((block_size, block_size), dtype=np.uint8)
            block = np.zeros((half_block_size, half_block_size), dtype=np.int16)
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
                    # ready for next iteration
                    mode_array_split.append(mode_split)
                    prediction_block_split[slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size] = \
                        np.add(prediction_sub_block, itran_split).clip(0, 255).astype(np.uint8)

                mode_array += mode_array_split
                pred[i][j] = prediction_block_split
                entropy_encode_str += code_str_split
    # do encoding and bit counting

    return pred, mode_array, split_array, entropy_encode_str
