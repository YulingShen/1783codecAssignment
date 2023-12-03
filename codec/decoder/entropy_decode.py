import numpy as np


def decode_quan_one_frame(code_str, n_w, n_h, block_len):
    block_size = block_len * block_len
    quan_frame = np.zeros((n_h, n_w, block_len, block_len))
    for i in range(n_h):
        for j in range(n_w):
            val_array, code_str = get_size(code_str, block_size)
            block = de_RLE(val_array, block_len)
            quan_frame[i][j] = block
    return quan_frame, code_str


def decode_quan_one_frame_VBS(code_str, n_w, n_h, block_len, split_array):
    block_size = block_len * block_len
    half_block_len = int(block_len / 2)
    sub_block_size = half_block_len * half_block_len
    quan_frame = np.zeros((n_h, n_w, block_len, block_len))
    for i in range(n_h):
        for j in range(n_w):
            split_mode = split_array[i * n_w + j]
            if split_mode == 0:
                val_array, code_str = get_size(code_str, block_size)
                block = de_RLE(val_array, block_len)
                quan_frame[i][j] = block
            else:
                single_block = np.zeros((block_len, block_len))
                for k in range(4):
                    slice_x = (k // 2) * half_block_len
                    slice_y = (k % 2) * half_block_len
                    val_array, code_str = get_size(code_str, sub_block_size)
                    block = de_RLE(val_array, half_block_len)
                    single_block[slice_x:slice_x + half_block_len, slice_y:slice_y + half_block_len] = block
                quan_frame[i][j] = single_block
    return quan_frame, code_str


def decode_quan_frame_VBS_given_row(code_str, n_w, block_len, split_array, quan_frame, i):
    block_size = block_len * block_len
    half_block_len = int(block_len / 2)
    sub_block_size = half_block_len * half_block_len
    for j in range(n_w):
        split_mode = split_array[j]
        if split_mode == 0:
            val_array, code_str = get_size(code_str, block_size)
            block = de_RLE(val_array, block_len)
            quan_frame[i][j] = block
        else:
            single_block = np.zeros((block_len, block_len))
            for k in range(4):
                slice_x = (k // 2) * half_block_len
                slice_y = (k % 2) * half_block_len
                val_array, code_str = get_size(code_str, sub_block_size)
                block = de_RLE(val_array, half_block_len)
                single_block[slice_x:slice_x + half_block_len, slice_y:slice_y + half_block_len] = block
            quan_frame[i][j] = single_block
    return quan_frame, code_str


def decode_quan_frame_VBS_given_block(code_str, block_len, split_array, quan_frame, i, j):
    block_size = block_len * block_len
    half_block_len = int(block_len / 2)
    sub_block_size = half_block_len * half_block_len
    split_mode = split_array[0]
    if split_mode == 0:
        val_array, code_str = get_size(code_str, block_size)
        block = de_RLE(val_array, block_len)
        quan_frame[i][j] = block
    else:
        single_block = np.zeros((block_len, block_len))
        for k in range(4):
            slice_x = (k // 2) * half_block_len
            slice_y = (k % 2) * half_block_len
            val_array, code_str = get_size(code_str, sub_block_size)
            block = de_RLE(val_array, half_block_len)
            single_block[slice_x:slice_x + half_block_len, slice_y:slice_y + half_block_len] = block
        quan_frame[i][j] = single_block
    return quan_frame, code_str

def decode_vec_one_frame(code_str, size, mv=True):
    if mv:
        x_array, code_str = get_size(code_str, size)
        y_array, code_str = get_size(code_str, size)
        k_array, code_str = get_size(code_str, size)
        # get size does not pad 0
        while len(x_array) < size:
            x_array.append(0)
        while len(y_array) < size:
            y_array.append(0)
        while len(k_array) < size:
            k_array.append(0)
        val_array = np.array([x_array, y_array, k_array]).T
    else:
        val_array, code_str = get_size(code_str, size)
        while len(val_array) < size:
            val_array.append(0)
    return val_array, code_str


def decode_vec_one_frame_alter(code_str, size, mv=True):
    if mv:
        vec_connect, code_str = get_size(code_str, size * 3)
        while len(vec_connect) < size * 3:
            vec_connect.append(0)
        val_array = []
        for i in range(size):
            val_array.append(vec_connect[i * 3: i * 3 + 3])
    else:
        val_array, code_str = get_size(code_str, size)
        while len(val_array) < size:
            val_array.append(0)
    return val_array, code_str


def decode_split_one_frame(code_str, size):
    split_diff, vec_code = get_size(code_str, size)
    while len(split_diff) < size:
        split_diff.append(0)
    return split_diff, vec_code


def decode_setting(code_str):
    values = []
    while len(code_str) > 0 and code_str[0] in ['0', '1']:
        zero_count = 0
        while code_str[zero_count] == "0":
            zero_count += 1
        bit_str = code_str[: zero_count * 2 + 1]
        code_str = code_str[zero_count * 2 + 1:]
        val = golomb_decode(bit_str)
        values.append(val)
    if values[5] == 1:
        VBSEnable = True
    else:
        VBSEnable = False
    if values[6] == 1:
        FMEEnable = True
    else:
        FMEEnable = False
    return values[0], values[1], values[2], values[3], values[4], VBSEnable, FMEEnable, values[7], values[8]


def get_size(code_str, size):
    remain_count = size
    val_array = []
    num_count = 0
    while remain_count > 0:
        zero_count = 0
        while code_str[zero_count] == "0":
            zero_count += 1
        bit_str = code_str[: zero_count * 2 + 1]
        code_str = code_str[zero_count * 2 + 1:]
        val = golomb_decode(bit_str)
        if num_count == 0:
            if val == 0:
                remain_count = 0
            elif val > 0:
                remain_count -= val
                for x in range(val):
                    val_array.append(0)
            else:
                num_count = -val
        else:
            num_count -= 1
            remain_count -= 1
            val_array.append(val)
    return val_array, code_str


def get_num(code_str, size):
    val_array = []
    while size > 0:
        zero_count = 0
        while code_str[zero_count] == "0":
            zero_count += 1
        bit_str = code_str[: zero_count * 2 + 1]
        code_str = code_str[zero_count * 2 + 1:]
        val = golomb_decode(bit_str)
        val_array.append(val)
        size -= 1
    return val_array, code_str

def de_RLE(num_array, block_len):
    result = np.zeros((block_len, block_len))
    x = 0
    y = 0
    for each in num_array:
        result[x][y] = each
        if y == 0 or x == block_len - 1:
            x = x + y
            y = min(block_len - 1, x + 1)
            x = x + 1 - y
        else:
            x += 1
            y -= 1
    return result


# decode a single number
def golomb_decode(bit_string):
    rep = int(bit_string[int(len(bit_string) / 2):], 2) - 1
    if rep % 2 == 0:
        return int(-rep / 2)
    return int((rep + 1) / 2)
