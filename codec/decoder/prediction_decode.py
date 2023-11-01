import numpy as np
from codec import blocking


def decode_residual_ME(prediction_array, residual, vector_array, w, h, block_size):
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
    # here depends on the input type
    # if both are uint8 it will round itself up
    # if being uint8 and int8, which is pred and signed residual, it will clip then transform to uint8
    return np.add(ME_prediction, residual).clip(0, 255).astype(np.uint8)


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
