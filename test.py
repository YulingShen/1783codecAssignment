import configparser

import numpy as np

from codec import quantization, blocking, evaluation
from codec.decoder import entropy_decode, differential_decode, prediction_decode, quantization_decode, transform_decode
from utils import reader
from codec.encoder import entropy_encode

if __name__ == '__main__':
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # b = np.tile(a[:, 4 - 1], (4, 1)).transpose()
    # c = np.tile(a[3, :], (4, 1))
    # mat = np.load('./files/psnr_arr.npy')
    # mat2 = np.load('./files/split_arr.npy')
    # mat3 = np.load('./files/bits_arr.npy')
    # print(mat)

    a = np.array([1,2,3,4])
    b = np.array([1,2,3,4])
    print(evaluation.calculate_psnr(a, b))