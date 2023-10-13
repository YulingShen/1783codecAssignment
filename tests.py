import numpy as np
import math
from codec import quantization
from codec.encoder import transform_encode, quantization_encode
from codec.decoder import transform_decode, quantization_decode


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
    mat = np.array([1,2,3,4,5,6,7,8])
    ee = np.tile(mat, (8, 1))
    aa = ee[:, 0]
    print(np.tile(ee[:, 0], (8, 1)))
