import numpy as np
from codec import blocking


def decode_residual_ME(prediction, residual, vector_array, w, h, block_size):
    n_w = (w - 1) // block_size + 1
    n_h = (h - 1) // block_size + 1
    block_prediction = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    for i in range(n_h):
        for j in range(n_w):
            vector = vector_array[i * n_w + j]
            block_prediction[i][j] = prediction[i * block_size + vector[0]: i * block_size + vector[0] + block_size,
                                     j * block_size + vector[1]: j * block_size + vector[1] + block_size]
    ME_prediction = blocking.deblock_frame(block_prediction, w, h)
    # here depends on the input type
    # if both are uint8 it will round itself up
    # if being uint8 and int8, which is pred and signed residual, it will clip then transform to uint8
    return np.add(ME_prediction, residual).clip(0, 255).astype(np.uint8)

def intra_decode(residual, mode_array, w, h, block_size):
    n_w = (w - 1) // block_size + 1
    n_h = (h - 1) // block_size + 1
    block_prediction = np.zeros((n_h, n_w, block_size, block_size), dtype=np.uint8)
    residual_block = blocking.block_frame(residual, block_size)
    for i in range(n_h):
        for j in range(n_w):
            mode = mode_array[i * n_w + j]
            # if mode == 0 and j != 0:
