import numpy as np


def quantization_block(frame_block, q):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    quan_block = np.zeros((n_h, n_w, 8, 8), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            quan = np.divide(frame_block[x][y], q)
            quan_block[x][y] = np.rint(quan)
    return quan_block
