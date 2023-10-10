import numpy as np
from codec import blocking


def closest_multi_power2(x, n):
    # special case, cannot round 127 to 128, will cause y value exceed 8 bit range and become 0
    if x == 127:
        return x
    base = 2 ** n
    return (x + int(base / 2)) & (-base)


def generate_residual(prediction, frame_block, w, h, n):
    frame = blocking.deblock_frame(frame_block, w, h)
    residual = np.subtract(frame, prediction)
    for i in range(h):
        for j in range(w):
            residual[i][j] = closest_multi_power2(residual[i][j], n)
    return residual


def find_predict_block(min_MAE, min_x, min_y, MAE, x, y):
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
    block_residual = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
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
                    diff = np.subtract(frame_block[i][j], pred).astype(np.int8)
                    MAE = np.sum(np.abs(diff))
                    min_MAE, min_x, min_y, changed = find_predict_block(min_MAE, min_x, min_y, MAE, x, y)
                    if changed:
                        block = diff
            MAE_sum += min_MAE
            for x in range(block_size):
                for y in range(block_size):
                    block[x][y] = closest_multi_power2(block[x][y], n)
            block_residual[i][j] = block
            vector_array.append([min_x, min_y])
    residual = blocking.deblock_frame(block_residual, w, h)
    return residual, vector_array, MAE_sum / (n_h * n_w)
