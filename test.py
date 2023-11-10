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
    mat = np.load('./files/psnr_arr.npy')
    mat2 = np.load('./files/split_arr.npy')
    mat3 = np.load('./files/bits_arr.npy')
    print(mat)

    # a = np.array([1,2,3,4])
    # b = np.array([1,2,3,4])
    # print(evaluation.calculate_psnr(a, b))

    # print('a')
    # a = reader.y_only_byte_frame_array(reader.read_raw_byte_array('./files/foreman_cif_y.yuv'), 352, 288)
    # print('b')
    # b = reader.y_only_byte_frame_array(reader.read_raw_byte_array('./files/foreman_cif_y_recon1_2.0.yuv'), 352, 288)
    # for i in range(10):
    #     print(evaluation.calculate_psnr(a[i], b[i]))


