import numpy as np


def dequantization_frame(frame_block, q):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    dequan_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            dequan_block[x][y] = dequantization_block(frame_block[x][y], q)
    return dequan_block


def dequantization_block(block, q):
    dequan = np.multiply(block, q)
    return np.rint(dequan)
