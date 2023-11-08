import numpy as np

from codec import quantization, blocking
from codec.decoder import entropy_decode, differential_decode, prediction_decode, quantization_decode, transform_decode
from utils import reader
from codec.encoder import entropy_encode

if __name__ == '__main__':
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
    # b = np.tile(a[:, 4 - 1], (4, 1)).transpose()
    # c = np.tile(a[3, :], (4, 1))
    with open('./files/test.npy', 'rb') as f:
        quan = np.load(f)
    with open('./files/test2.npy', 'rb') as f:
        itransform = np.load(f)
    with open('./files/foreman_cif_y_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, i, qp, period, VBSEnable, FMEEnable = entropy_decode.decode_setting(setting)
    q = quantization.generate_q(i, qp)
    if qp > 0:
        q_split = quantization.generate_q(int(i / 2), qp - 1)
    else:
        q_split = quantization.generate_q(int(i / 2), qp)
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1
    split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_block_h * n_block_w)
    split_array = differential_decode.differential_decode(split_diff)
    len_array = n_block_h * n_block_w + 3 * np.sum(split_array)
    vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, False)
    mode_array = differential_decode.differential_decode(vec_diff)

    dequan = quantization_decode.dequantization_frame_VBS(quan, q, q_split, split_array)
    itran = transform_decode.inverse_transform_frame_VBS(dequan, split_array)
    res = blocking.deblock_frame(itransform, w, h)
    recon = prediction_decode.intra_decode_VBS(res, mode_array, split_array, w, h, i)
    reader.write_frame_array_to_file([recon], './files/foreman_cif_y_recon.yuv')
