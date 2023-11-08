import numpy as np


def quantization_frame(frame_block, q):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    quan_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            quan_block[x][y] = quantization_block(frame_block[x][y], q)
    return quan_block


def quantization_block(block, q):
    quan = np.divide(block, q)
    return np.rint(quan)
