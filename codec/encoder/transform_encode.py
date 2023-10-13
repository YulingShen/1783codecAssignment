from codec import dct_2d
import numpy as np


def transform_block(frame_block):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    dct_block = np.zeros((n_h, n_w, 8, 8), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            dct = dct_2d.dct_2d(frame_block[x][y])
            dct_block[x][y] = np.rint(dct)
    return dct_block
