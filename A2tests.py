import configparser

import numpy as np

from codec import A2process, evaluation
from codec.decoder import entropy_decode, differential_decode
from utils import reader

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    w = config_dict['w']
    h = config_dict['h']
    i = config_dict['i']
    n = config_dict['n']
    r = config_dict['r']
    qp = config_dict['qp']
    period = config_dict['period']
    frame = config_dict['frame']
    VBSEnable = config_dict['VBSEnable']
    nRefFrames = config_dict['nRefFrames']
    VBSEnable = config_dict['VBSEnable']
    # this is the coefficient to adjust lambda value
    lambda_coefficient = config_dict['lambda_coefficient']
    FMEEnable = config_dict['FMEEnable']
    FastME = config_dict['FastME']
    config = configparser.ConfigParser()
    config.read("./test_config.yaml")
    origin_array = reader.read_raw_byte_array(filepath)
    psnr_array = []
    split_rate_array = []
    bits_array = []
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1
    for qp in [1, 4, 7, 10]:
        for section in config.sections():
            lambda_coefficient = float(config[section]['lambda_coefficient'])
            print('case qp ' + str(qp) + ' lambda ' + str(lambda_coefficient))
            recon_array, bits = A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable,
                                                    lambda_coefficient, FMEEnable, FastME, frame)
            bits_array.append(bits)
            psnr = []
            for num in range(frame):
                each_psnr = evaluation.calculate_psnr(origin_array[num], recon_array[num])
                psnr.append(each_psnr)
            psnr_array.append(psnr)
            split_rate = []
            with open(filepath[:-4] + '_diff', 'r') as vec_file:
                setting = vec_file.readline()
                vec_code = vec_file.read()
            frame_count = 0
            while len(vec_code) > 0:
                split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_block_h * n_block_w)
                split_array = differential_decode.differential_decode(split_diff)
                len_array = n_block_h * n_block_w + 3 * np.sum(split_array)
                if frame_count % period == 0:
                    vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, False)
                else:
                    vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, True)
                frame_count += 1
                split_rate.append(np.sum(split_array) / len(split_array))
            split_rate_array.append(split_rate)
    np.save('./files/psnr_arr.npy', psnr_array)
    np.save('./files/split_arr.npy', split_rate_array)
    np.save('./files/bits_arr.npy', np.array(bits_array))
