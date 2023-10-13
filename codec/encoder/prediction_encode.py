import numpy as np
from codec import blocking


def closest_multi_power2(x, n):
    base = 2 ** n
    return (x + int(base / 2)) & (-base)


def round_down_multi_power2(x, n):
    return x & -(2 ** n)


def generate_residual(prediction, frame_block, w, h, n):
    frame = blocking.deblock_frame(frame_block, w, h)
    residual = np.subtract(frame.astype(np.int16), prediction.astype(np.int16)).astype(np.int8)
    for i in range(h):
        for j in range(w):
            residual[i][j] = round_down_multi_power2(residual[i][j], n)
    return residual.astype(np.uint8)


def compare_MAE(min_MAE, min_x, min_y, MAE, x, y):
    if min_MAE < MAE:
        return min_MAE, min_x, min_y, False
    elif min_MAE > MAE:
        return MAE, x, y, True
    if abs(min_x) + abs(min_y) < abs(x) + abs(y):
        return min_MAE, min_x, min_y, False
    elif abs(min_x) + abs(min_y) > abs(x) + abs(y):
        return MAE, x, y, True
    if min_y < y:
        return min_MAE, min_x, min_y, False
    elif min_y > y:
        return MAE, x, y, True
    if min_x < x:
        return min_MAE, min_x, min_y, False
    return MAE, x, y, True


def generate_residual_ME(prediction, frame_block, w, h, n, r):
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
            min_MAE = 256 * block_size * block_size
            min_x = r + 1
            min_y = r + 1
            block = []
            for x in range(-r, r + 1):
                if x + i_h < 0 or x + i_h + block_size > h:
                    continue
                for y in range(-r, r + 1):
                    if y + i_w < 0 or y + i_w + block_size > w:
                        continue
                    pred = prediction[x + i_h: x + i_h + block_size, y + i_w: y + i_w + block_size]
                    diff = np.subtract(frame_block[i][j].astype(np.int16), pred.astype(np.int16))
                    MAE = np.sum(np.abs(diff))
                    min_MAE, min_x, min_y, changed = compare_MAE(min_MAE, min_x, min_y, MAE, x, y)
                    if changed:
                        block = diff
            MAE_sum += min_MAE
            block = block.astype(np.int8)
            for x in range(block_size):
                for y in range(block_size):
                    block[x][y] = round_down_multi_power2(block[x][y], n)
            block_residual[i][j] = block
            vector_array.append([min_x, min_y])
    block_residual = block_residual.astype(np.uint8)
    return block_residual, vector_array, MAE_sum / (h * w)


def intra_residual(frame_block, n):
    block_size = len(frame_block[0][0])
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    # 0 for horizontal, 1 for vertical
    mode_array = []
    block_residual = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    pred = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
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
            diff = diff.astype(np.int8)
            for x in range(block_size):
                for y in range(block_size):
                    diff[x][y] = round_down_multi_power2(diff[x][y], n)
            block_residual[i][j] = diff
            pred[i][j] = np.add(prediction_block, diff)
    block_residual = block_residual.astype(np.uint8)
    return block_residual, pred
