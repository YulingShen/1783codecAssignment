import numpy as np

from codec import evaluation
from codec.decoder import quantization_decode, transform_decode
from codec.encoder import transform_encode, quantization_encode, entropy_encode


def closest_multi_power2(x, n):
    base = 2 ** n
    return (x + int(base / 2)) & (-base)


def round_down_multi_power2(x, n):
    return x & -(2 ** n)


def compare_MAE(min_MAE, min_x, min_y, min_k, MAE, x, y, k):
    if min_MAE < MAE:
        return min_MAE, min_x, min_y, min_k, False
    elif min_MAE > MAE:
        return MAE, x, y, k, True
    if abs(min_x) + abs(min_y) < abs(x) + abs(y):
        return min_MAE, min_x, min_y, min_k, False
    elif abs(min_x) + abs(min_y) > abs(x) + abs(y):
        return MAE, x, y, k, True
    if min_y < y:
        return min_MAE, min_x, min_y, min_k, False
    elif min_y > y:
        return MAE, x, y, k, True
    if min_x < x:
        return min_MAE, min_x, min_y, min_k, False
    return MAE, x, y, k, True


def generate_residual_ME(prediction_array, frame_block, w, h, n, r, FMEEnable, FastME):
    block_size = len(frame_block[0][0])
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    vector_array = []
    block_residual = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for i in range(n_h):
        mvp = [0, 0]
        for j in range(n_w):
            # top left of the block
            i_h = i * block_size
            i_w = j * block_size
            if not FMEEnable:
                min_MAE, block, vec = search_motion_non_fraction(w, h, i_h, i_w, block_size, r, prediction_array,
                                                                 frame_block[i][j], FastME, mvp)
            else:
                min_MAE, block, vec = search_motion_fraction(w, h, i_h, i_w, block_size, r, prediction_array,
                                                             frame_block[i][j], FastME, mvp)
            for x in range(block_size):
                for y in range(block_size):
                    block[x][y] = closest_multi_power2(block[x][y], n)
            block_residual[i][j] = block
            vector_array.append(vec)
            mvp = vec[:2]
    return block_residual, vector_array


def generate_residual_ME_VBS(prediction_array, frame_block, w, h, n, r, lambda_val, q_non_split, q_split, FMEEnable, FastME):
    block_size = len(frame_block[0][0])
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    vector_array = []
    split_array = []
    half_block_size = int(block_size / 2)
    code_str_entropy = ''
    prev_vec = [0, 0, 0]
    prev_split = 0
    block_itran = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for i in range(n_h):
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
                                                                                slice_y:slice_y + half_block_size], FastME, mvp_split)
                    else:
                        min_MAE, block_origin, vec = search_motion_fraction(w, h, i_h, i_w, half_block_size, r,
                                                                            prediction_array,
                                                                            frame_block[i][j][
                                                                            slice_x:slice_x + half_block_size,
                                                                            slice_y:slice_y + half_block_size], FastME, mvp_split)
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


def search_motion_non_fraction(w, h, i_h, i_w, block_size, r, prediction_array, ref, FastME, mvp = None):
    # top left of the block
    min_MAE = 256 * block_size * block_size
    min_x = r + 1
    min_y = r + 1
    min_k = len(prediction_array) + 1
    # this is the residual
    block = []
    # full range search
    if not FastME:
        for x in range(-r, r + 1):
            if x + i_h < 0 or x + i_h + block_size > h:
                continue
            for y in range(-r, r + 1):
                if y + i_w < 0 or y + i_w + block_size > w:
                    continue
                for k in range(len(prediction_array)):
                    pred = prediction_array[k][x + i_h: x + i_h + block_size, y + i_w: y + i_w + block_size]
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    min_MAE, min_x, min_y, min_k, changed = compare_MAE(min_MAE, min_x, min_y, min_k, MAE, x, y, k)
                    if changed:
                        block = diff
    else:
        base_x = mvp[0]
        while base_x + i_h < 0:
            base_x += 1
        while base_x + i_h + block_size > h:
            base_x -= 1
        base_y = mvp[1]
        while base_y + i_w < 0:
            base_y += 1
        while base_y + i_w + block_size > w:
            base_y -= 1
        base_pair = [base_x, base_y]
        for k in range(len(prediction_array)):
            pred_frame = prediction_array[k]
            # get base case first
            base_x = base_pair[0]
            base_y = base_pair[1]
            pred = pred_frame[base_x + i_h: base_x + i_h + block_size, base_y + i_w: base_y + i_w + block_size]
            diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
            step_min_MAE = np.sum(np.abs(diff))
            search_signal = True
            while search_signal:
                origin_MAE = step_min_MAE
                origin_x = base_x
                origin_y = base_y
                for x_next in [-1, 1]:
                    x = origin_x + x_next
                    if x + i_h < 0 or x + i_h + block_size > h:
                        continue
                    y = origin_y
                    pred = pred_frame[x + i_h: x + i_h + block_size, y + i_w: y + i_w + block_size]
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    if MAE < step_min_MAE:
                        base_x = x
                        step_min_MAE = MAE
                for y_next in [-1, 1]:
                    y = origin_y + y_next
                    if y + i_w < 0 or y + i_w + block_size > w:
                        continue
                    x = origin_x
                    pred = pred_frame[x + i_h: x + i_h + block_size, y + i_w: y + i_w + block_size]
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    if MAE < step_min_MAE:
                        base_y = y
                        step_min_MAE = MAE
                if step_min_MAE >= origin_MAE:
                    search_signal = False
            min_MAE, min_x, min_y, min_k, changed = compare_MAE(min_MAE, min_x, min_y, min_k, step_min_MAE, base_x, base_y, k)
            if changed:
                block = diff
    return min_MAE, block, [min_x, min_y, min_k]


def search_motion_fraction(w, h, i_h, i_w, block_size, r, prediction_array, ref, FastME, mvp = None):
    # top left of the block
    min_MAE = 256 * block_size * block_size
    min_x = 2 * r + 1
    min_y = 2 * r + 1
    min_k = len(prediction_array) + 1
    block = []
    # full range search
    if not FastME:
        for x in range(-2 * r, 2 * r + 1):
            if x / 2 + i_h < 0 or x / 2 + i_h + block_size > h:
                continue
            for y in range(-2 * r, 2 * r + 1):
                if y / 2 + i_w < 0 or y / 2 + i_w + block_size > w:
                    continue
                if x % 2 == 0:
                    x_arr = [x // 2]
                else:
                    x_arr = [x // 2, (x + 1) // 2]
                if y % 2 == 0:
                    y_arr = [y // 2]
                else:
                    y_arr = [y // 2, (y + 1) // 2]
                for k in range(len(prediction_array)):
                    pred = np.zeros((block_size, block_size), dtype=np.int16)
                    block_count = 0
                    for each_x in x_arr:
                        for each_y in y_arr:
                            pred = np.add(pred, prediction_array[k][each_x + i_h: each_x + i_h + block_size,
                                                each_y + i_w: each_y + i_w + block_size])
                            block_count += 1
                    pred = pred // block_count
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    min_MAE, min_x, min_y, min_k, changed = compare_MAE(min_MAE, min_x, min_y, min_k, MAE, x, y, k)
                    if changed:
                        block = diff
    else:
        base_x = mvp[0]
        while base_x / 2 + i_h < 0:
            base_x += 1
        while base_x / 2 + i_h + block_size > h:
            base_x -= 1
        base_y = mvp[1]
        while base_y / 2 + i_w < 0:
            base_y += 1
        while base_y / 2 + i_w + block_size > w:
            base_y -= 1
        base_pair = [base_x, base_y]
        for k in range(len(prediction_array)):
            pred_frame = prediction_array[k]
            # get base case first
            base_x = base_pair[0]
            base_y = base_pair[1]
            if base_x % 2 == 0:
                x_arr = [base_x // 2]
            else:
                x_arr = [base_x // 2, (base_x + 1) // 2]
            if base_y % 2 == 0:
                y_arr = [base_y // 2]
            else:
                y_arr = [base_y // 2, (base_y + 1) // 2]
            pred = np.zeros((block_size, block_size), dtype=np.int16)
            block_count = 0
            for each_x in x_arr:
                for each_y in y_arr:
                    pred = np.add(pred, pred_frame[each_x + i_h: each_x + i_h + block_size,
                                        each_y + i_w: each_y + i_w + block_size])
                    block_count += 1
            pred = pred // block_count
            diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
            step_min_MAE = np.sum(np.abs(diff))
            search_signal = True
            while search_signal:
                origin_MAE = step_min_MAE
                origin_x = base_x
                origin_y = base_y
                for x_next in [-1, 1]:
                    x = origin_x + x_next
                    if x / 2 + i_h < 0 or x / 2 + i_h + block_size > h:
                        continue
                    y = origin_y
                    if base_x % 2 == 0:
                        x_arr = [x // 2]
                    else:
                        x_arr = [x // 2, (x + 1) // 2]
                    if base_y % 2 == 0:
                        y_arr = [y // 2]
                    else:
                        y_arr = [y // 2, (y + 1) // 2]
                    pred = np.zeros((block_size, block_size), dtype=np.int16)
                    block_count = 0
                    for each_x in x_arr:
                        for each_y in y_arr:
                            pred = np.add(pred, pred_frame[each_x + i_h: each_x + i_h + block_size,
                                                each_y + i_w: each_y + i_w + block_size])
                            block_count += 1
                    pred = pred // block_count
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    if MAE < step_min_MAE:
                        base_x = x
                        step_min_MAE = MAE
                for y_next in [-1, 1]:
                    y = origin_y + y_next
                    if y / 2 + i_w < 0 or y / 2 + i_w + block_size > w:
                        continue
                    x = origin_x
                    if base_x % 2 == 0:
                        x_arr = [x // 2]
                    else:
                        x_arr = [x // 2, (x + 1) // 2]
                    if base_y % 2 == 0:
                        y_arr = [y // 2]
                    else:
                        y_arr = [y // 2, (y + 1) // 2]
                    pred = np.zeros((block_size, block_size), dtype=np.int16)
                    block_count = 0
                    for each_x in x_arr:
                        for each_y in y_arr:
                            pred = np.add(pred, pred_frame[each_x + i_h: each_x + i_h + block_size,
                                                each_y + i_w: each_y + i_w + block_size])
                            block_count += 1
                    pred = pred // block_count
                    diff = np.subtract(ref.astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    if MAE < step_min_MAE:
                        base_y = y
                        step_min_MAE = MAE
                if step_min_MAE >= origin_MAE:
                    search_signal = False
            min_MAE, min_x, min_y, min_k, changed = compare_MAE(min_MAE, min_x, min_y, min_k, step_min_MAE, base_x,
                                                                base_y, k)
            if changed:
                block = diff
    return min_MAE, block, [min_x, min_y, min_k]


def intra_residual(frame_block, n, q):
    block_size = len(frame_block[0][0])
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    # 0 for horizontal, 1 for vertical
    mode_array = []
    block_residual = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    pred = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    quan_frame = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for i in range(n_h):
        for j in range(n_w):
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
                mode_array.append(1)
                diff = diff_ver
                prediction_block = prediction_block_ver
            else:
                mode_array.append(0)
                diff = diff_hor
                prediction_block = prediction_block_hor
            for x in range(block_size):
                for y in range(block_size):
                    diff[x][y] = closest_multi_power2(diff[x][y], n)
            block_residual[i][j] = diff
            tran = transform_encode.transform_block(diff)
            quan = quantization_encode.quantization_block(tran, q)
            quan_frame[i][j] = quan
            dequan = quantization_decode.dequantization_block(quan, q)
            itran = transform_decode.inverse_transform_block(dequan)
            pred[i][j] = np.add(prediction_block, itran).clip(0, 255).astype(np.uint8)
    return block_residual, pred, mode_array, quan_frame


def intra_residual_VBS(frame_block, n, lambda_val, q_non_split, q_split):
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    # 0 for horizontal, 1 for vertical
    mode_array = []
    split_array = []
    prev_mode = 0
    prev_split = 0
    entropy_encode_str = ''
    pred = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    blank_half = np.full((half_block_size, half_block_size), 128, dtype=np.int16)
    for i in range(n_h):
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
    return pred, mode_array, split_array, entropy_encode_str
