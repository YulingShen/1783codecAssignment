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
    return np.add(ME_prediction, residual)
