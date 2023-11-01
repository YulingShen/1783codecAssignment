import numpy as np


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


def generate_residual_ME(prediction_array, frame_block, w, h, n, r):
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
                        diff = np.subtract(frame_block[i][j].astype(np.int16), pred.astype(np.int16))
                        MAE = np.sum(np.abs(diff))
                        min_MAE, min_x, min_y, min_k, changed = compare_MAE(min_MAE, min_x, min_y, min_k, MAE, x, y, k)
                        if changed:
                            block = diff
            MAE_sum += min_MAE
            for x in range(block_size):
                for y in range(block_size):
                    block[x][y] = closest_multi_power2(block[x][y], n)
            block_residual[i][j] = block
            vector_array.append([min_x, min_y, min_k])
    return block_residual, vector_array, MAE_sum / (h * w)
