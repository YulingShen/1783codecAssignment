import numpy as np


def dequantization_block(frame_block, q):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    dequan_block = np.zeros((n_h, n_w, 8, 8), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            dequan = np.multiply(frame_block[x][y], q)
            dequan_block[x][y] = np.rint(dequan)
    return dequan_block
