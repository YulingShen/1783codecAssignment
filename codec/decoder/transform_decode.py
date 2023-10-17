from codec import dct_2d
import numpy as np


def inverse_transform_frame(frame_block):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    idct_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            idct_block[x][y] = inverse_transform_block(frame_block[x][y])
    return idct_block


def inverse_transform_block(block):
    dct = dct_2d.idct_2d(block)
    return np.rint(dct)
