def entropy_encode_quan_frame_block(quan_frame_block):
    n_h = len(quan_frame_block)
    n_w = len(quan_frame_block[0])
    block_size = len(quan_frame_block[0][0])
    coded_string = ""
    bit_sum = 0
    for i in range(n_h):
        for j in range(n_w):
            block = quan_frame_block[i][j]
            num_array = []
            for xy_sum in range(block_size * 2 - 1):
                for x in range(max(0, xy_sum - block_size + 1), min(xy_sum + 1, block_size)):
                    y = xy_sum - x
                    num_array.append(block[x][y])
            rle_array = RLE(num_array)
            for each in rle_array:
                code, bits = exp_golomb(each)
                coded_string = coded_string + code
                bit_sum = bit_sum + bits
    return coded_string, bit_sum


def entropy_encode_vec(vector_array):
    result = ""
    bit_sum = 0
    if len(vector_array.shape) == 2:
        for d in range(vector_array.shape[1]):
            arr = vector_array[:, d]
            rle = RLE(arr)
            for each in rle:
                code, bits = exp_golomb(each)
                result = result + code
                bit_sum = bit_sum + bits
    else:
        rle = RLE(vector_array)
        for each in rle:
            code, bits = exp_golomb(each)
            result = result + code
            bit_sum = bit_sum + bits
    return result, bit_sum


def entropy_encode_setting(w, h, i, qp, period):
    result = ""
    bit_sum = 0
    code, bits = exp_golomb(w)
    result += code
    bit_sum += bits
    code, bits = exp_golomb(h)
    result += code
    bit_sum += bits
    code, bits = exp_golomb(i)
    result += code
    bit_sum += bits
    code, bits = exp_golomb(qp)
    result += code
    bit_sum += bits
    code, bits = exp_golomb(period)
    result += code
    bit_sum += bits
    return result, bit_sum


def RLE(num_array):
    count = 0
    count_index = 0
    result = []
    if num_array[0] == 0:
        zero = False
    else:
        zero = True
    for each in num_array:
        if zero and each != 0:
            if count != 0:
                result[count_index] = count
            count_index = len(result)
            result.append(0)
            zero = False
            count = 0
        elif not zero and each == 0:
            if count != 0:
                result[count_index] = -count
            count_index = len(result)
            result.append(0)
            zero = True
            count = 0
        if not zero:
            result.append(each)
        count += 1
    if count != 0 and not zero:
        result[count_index] = -count
    return result


def exp_golomb(val):
    if val <= 0:
        rep = abs(val) * 2
    else:
        rep = val * 2 - 1
    sec_half = bin(rep + 1)[2:]
    fst_half = "0" * (len(sec_half) - 1)
    return fst_half + sec_half, 2 * len(sec_half) - 1
