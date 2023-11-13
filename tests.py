import numpy as np
import math
from codec import quantization
from codec.encoder import transform_encode, quantization_encode, entropy_encode
from codec.decoder import transform_decode, quantization_decode, entropy_decode
from utils import reader


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
    reader.res_abs("./files/foreman_cif_y_res_ME.yuv")