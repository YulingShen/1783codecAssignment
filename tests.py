import numpy as np
import math
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
    return ((x + int(base / 2)) & (-base))


if __name__ == '__main__':
    # bytes = reader.read_raw_byte_array('./files/foreman_cif_res.yuv')
    # also_bytes = reader.read_raw_byte_array('./files/foreman_cif_recon.yuv')
    # frames = reader.y_only_byte_frame_array(bytes, 352, 288)
    # print(frames)
    vecs = np.load('./files/test.npy')
    print(vecs)
