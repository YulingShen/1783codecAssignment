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


def dequantization_frame_VBS(frame_block, q, q_split, split_array):
    n_h = len(frame_block)
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    dequan_block = np.zeros((n_h, n_w, block_size, block_size), dtype=np.int16)
    for x in range(n_h):
        for y in range(n_w):
            split_mode = split_array[x * n_w + y]
            if split_mode == 0:
                dequan_block[x][y] = dequantization_block(frame_block[x][y], q)
            else:
                single_block = np.zeros((block_size, block_size))
                for k in range(4):
                    slice_x = (k // 2) * half_block_size
                    slice_y = (k % 2) * half_block_size
                    single_block[slice_x:slice_x + half_block_size,
                    slice_y:slice_y + half_block_size] = dequantization_block(
                        frame_block[x][y][slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size],
                        q_split)
                dequan_block[x][y] = single_block
    return dequan_block

def dequantization_frame_VBS_given_row(frame_block, q, q_split, split_array, dequan_block, i):
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    for y in range(n_w):
        split_mode = split_array[y]
        if split_mode == 0:
            dequan_block[i][y] = dequantization_block(frame_block[i][y], q)
        else:
            single_block = np.zeros((block_size, block_size))
            for k in range(4):
                slice_x = (k // 2) * half_block_size
                slice_y = (k % 2) * half_block_size
                single_block[slice_x:slice_x + half_block_size,
                slice_y:slice_y + half_block_size] = dequantization_block(
                    frame_block[i][y][slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size],
                    q_split)
            dequan_block[i][y] = single_block
    return dequan_block


def dequantization_frame_VBS_given_block(frame_block, q, q_split, split_array, dequan_block, i, j):
    n_w = len(frame_block[0])
    block_size = len(frame_block[0][0])
    half_block_size = int(block_size / 2)
    split_mode = split_array[0]
    if split_mode == 0:
        dequan_block[i][j] = dequantization_block(frame_block[i][j], q)
    else:
        single_block = np.zeros((block_size, block_size))
        for k in range(4):
            slice_x = (k // 2) * half_block_size
            slice_y = (k % 2) * half_block_size
            single_block[slice_x:slice_x + half_block_size,
            slice_y:slice_y + half_block_size] = dequantization_block(
                frame_block[i][j][slice_x:slice_x + half_block_size, slice_y:slice_y + half_block_size],
                q_split)
        dequan_block[i][j] = single_block
    return dequan_block

def dequantization_block(block, q):
    dequan = np.multiply(block, q)
    return np.rint(dequan)
