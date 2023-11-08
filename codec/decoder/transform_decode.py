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


def inverse_transform_frame_VBS(frame_block, split_array):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    idct_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            split_mode = split_array[x * n_w + y]
            if split_mode == 0:
                idct_block[x][y] = inverse_transform_block(frame_block[x][y])
            else:
                single_block = np.zeros((block_size, block_size))
                for k in range(4):
                    slice_x = (k // 2) * half_block_size
                    slice_y = (k % 2) * half_block_size
                    single_block[slice_x:slice_x + half_block_size,
                    slice_y:slice_y + half_block_size] = inverse_transform_block(
                        frame_block[x][y][slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size])
                idct_block[x][y] = single_block
    return idct_block


def inverse_transform_block(block):
    dct = dct_2d.idct_2d(block)
    return np.rint(dct)
