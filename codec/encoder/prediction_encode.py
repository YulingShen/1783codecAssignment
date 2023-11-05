import numpy as np

from utils import psnr
from codec.decoder import quantization_decode, transform_decode
from codec.encoder import transform_encode, quantization_encode


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


def generate_residual_ME(prediction_array, frame_block, w, h, n, r, FMEEnable):
    block_size = len(frame_block[0][0])
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    MAE_sum = 0
    vector_array = []
    block_residual = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for i in range(n_h):
        for j in range(n_w):
            # top left of the block
            i_h = i * block_size
            i_w = j * block_size
            if not FMEEnable:
                min_MAE, block, vec = search_motion_non_fraction(w, h, i_h, i_w, block_size, r, prediction_array,
                                                                 frame_block[i][j])
            else:
                min_MAE, block, vec = search_motion_fraction(w, h, i_h, i_w, block_size, r, prediction_array,
                                                             frame_block[i][j])
            MAE_sum += min_MAE
            for x in range(block_size):
                for y in range(block_size):
                    block[x][y] = closest_multi_power2(block[x][y], n)
            block_residual[i][j] = block
            vector_array.append(vec)
    return block_residual, vector_array, MAE_sum / (h * w)


def search_motion_non_fraction(w, h, i_h, i_w, block_size, r, prediction_array, ref):
    # top left of the block
    min_MAE = 256 * block_size * block_size
    min_x = r + 1
    min_y = r + 1
    min_k = len(prediction_array) + 1
    block = []
    # full range search
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
    return min_MAE, block, [min_x, min_y, min_k]


def search_motion_fraction(w, h, i_h, i_w, block_size, r, prediction_array, ref):
    # top left of the block
    min_MAE = 256 * block_size * block_size
    min_x = 2 * r + 1
    min_y = 2 * r + 1
    min_k = len(prediction_array) + 1
    block = []
    # full range search
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
