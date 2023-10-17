from codec import dct_2d
import numpy as np


def transform_frame(frame_block):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    dct_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            dct_block[x][y] = transform_block(frame_block[x][y])
    return dct_block


def transform_block(block):
    dct = dct_2d.dct_2d(block)
    return np.rint(dct)
