import numpy as np


def decode_quan_one_frame(code_str, n_w, n_h, block_len):
    block_size = block_len * block_len
    quan_frame = np.zeros((n_h, n_w, block_len, block_len))
    for i in range(n_h):
        for j in range(n_w):
            # print("block" + str(i) + ":" + str(j))
            val_array, code_str = get_size(code_str, block_size)
            block = de_RLE(val_array, block_len)
            quan_frame[i][j] = block
    return quan_frame, code_str


def decode_vec_one_frame(code_str, size, mv=True):
    if mv:
        x_array, code_str = get_size(code_str, size)
        y_array, code_str = get_size(code_str, size)
        while len(x_array) < size:
            x_array.append(0)
        while len(y_array) < size:
            y_array.append(0)
        val_array = np.array([x_array, y_array]).T
    else:
        val_array, code_str = get_size(code_str, size)
        while len(val_array) < size:
            val_array.append(0)
    return val_array, code_str


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
