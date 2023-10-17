import numpy as np
import math
from codec import quantization
from codec.encoder import transform_encode, quantization_encode, entropy_encode
from codec.decoder import transform_decode, quantization_decode, entropy_decode


def closest_power2(x):
    neg = 1
    if x < 0:
        neg = -1
        x = -x
    if x < 2:
        return neg * x
    op = math.floor if bin(x)[3] != "1" else math.ceil
    return neg * (2 ** (op(math.log(x, 2))))


def closest_mult(x, n):
    base = 2 ** n
    return (x + int(base / 2)) & (-base)


if __name__ == '__main__':
    mat = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]])
    print(entropy_encode.entropy_encode_quan_frame_block(mat))
