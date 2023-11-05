import numpy as np
from codec import blocking


def decode_residual_ME(prediction_array, residual, vector_array, w, h, block_size, FMEEnable):
    n_w = (w - 1) // block_size + 1
    n_h = (h - 1) // block_size + 1
    block_prediction = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            vector = vector_array[i * n_w + j]
            pred_frame = prediction_array[vector[2]]
            if not FMEEnable:
                block_prediction[i][j] = decode_res_block_non_fraction(pred_frame, vector, i, j, block_size)
            else:
                block_prediction[i][j] = decode_res_block_fraction(pred_frame, vector, i, j, block_size)
    ME_prediction = blocking.deblock_frame(block_prediction, w, h)
    # here depends on the input type
    # if both are uint8 it will round itself up
    # if being uint8 and int8, which is pred and signed residual, it will clip then transform to uint8
    return np.add(ME_prediction, residual).clip(0, 255).astype(np.uint8)


def decode_res_block_non_fraction(pred_frame, vector, i, j, block_size):
    return pred_frame[i * block_size + vector[0]: i * block_size + vector[0] + block_size,
           j * block_size + vector[1]: j * block_size + vector[1] + block_size]


def decode_res_block_fraction(pred_frame, vector, i, j, block_size):
    x = vector[0]
    y = vector[1]
    if x % 2 == 0:
        x_arr = [x // 2]
    else:
        x_arr = [x // 2, (x + 1) // 2]
    if y % 2 == 0:
        y_arr = [y // 2]
    else:
        y_arr = [y // 2, (y + 1) // 2]
    pred = np.zeros((block_size, block_size), dtype=np.int16)
    block_count = 0
    for each_x in x_arr:
        for each_y in y_arr:
            pred = np.add(pred, pred_frame[i * block_size + each_x: i * block_size + each_x + block_size,
                                j * block_size + each_y: j * block_size + each_y + block_size])
            block_count += 1
    pred = pred // block_count
    return pred


def generate_prediction(prediction_array, vector_array, w, h, block_size):
    n_w = (w - 1) // block_size + 1
    n_h = (h - 1) // block_size + 1
    block_prediction = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            vector = vector_array[i * n_w + j]
            block_prediction[i][j] = prediction_array[vector[2]][
                                     i * block_size + vector[0]: i * block_size + vector[0] + block_size,
                                     j * block_size + vector[1]: j * block_size + vector[1] + block_size]
    ME_prediction = blocking.deblock_frame(block_prediction, w, h)
    return ME_prediction.clip(0, 255).astype(np.uint8)


def intra_decode(residual, mode_array, w, h, block_size):
    n_w = (w - 1) // block_size + 1
    n_h = (h - 1) // block_size + 1
    result = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    residual_block = blocking.block_frame(residual, block_size)
    blank = np.full((block_size, block_size), 128, dtype=np.int16)
    for i in range(n_h):
        for j in range(n_w):
            mode = mode_array[i * n_w + j]
            if mode == 0 and j != 0:
                prev = result[i][j - 1]
                prediction_block = np.tile(prev[:, block_size - 1], (block_size, 1)).transpose()
            elif mode == 1 and i != 0:
                prev = result[i - 1][j]
                prediction_block = np.tile(prev[block_size - 1], (block_size, 1))
            else:
                prediction_block = blank
            result[i][j] = np.add(prediction_block, residual_block[i][j]).clip(0, 255).astype(np.uint8)
    return blocking.deblock_frame(result, w, h)
