import time

import configparser

import numpy as np

from codec import A3process, evaluation
from utils import reader

def DELIVER_EX1():
    filename_CIF = "./files/CIF_y"
    filepath_CIF = filename_CIF+'.yuv'

    filename_QCIF = "./files/QCIF_y"
    filepath_QCIF = filename_QCIF+'.yuv'

    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    tablepath = "./rate_table.yaml"
    table_dict = reader.load_RC_profile(tablepath)
    w, h = 352, 288
    origin_array_cif = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath_CIF), w, h, 21)
    w_q, h_q = 352//2, 288//2
    origin_array_qcif = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath_QCIF), w_q, h_q, 21)
    for name in ["CIF", "QCIF"]:
        if name == "CIF":
            config_dict['w'], config_dict['h'] = w, h
            config_dict['targetBR'] = 2737152
        else:
            config_dict['w'], config_dict['h'] = w_q, h_q
            config_dict['targetBR'] = 1094860
        for period in [1,4,21]:
            if period == 21:
                config_dict['RCFlag'] = 0
                config_dict['qp'] = 5
            config_dict['period'] = period
            filepath = filepath_CIF if name == "CIF" else filepath_QCIF
            filename = filename_CIF if name == "CIF" else filename_QCIF
            recon_array, bit_count_arr = A3process.encode_complete(filepath, config_dict, table_dict)
            reader.write_frame_array_to_file(recon_array, filename +"_{}_".format(name) + str(period) + 'recon' + '.yuv')
            psnr = []
            for num in range(21):
                origin_array = origin_array_cif if name == "CIF" else origin_array_qcif
                each_psnr = round(evaluation.calculate_psnr(origin_array[num], recon_array[num]), 2)
                psnr.append(each_psnr)
            config_dict['RCFlag'] = 1
            print(name, period)
            print("BITS\n", bit_count_arr)
            print("PSNR\n", psnr)

def DELIVER_EX1_2():
    filepath = './files/CIF_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    tablepath = "./rate_table.yaml"
    table_dict = reader.load_RC_profile(tablepath)
    # w, h = config_dict['w'], config_dict['h']
    # origin_array = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath), w, h, frame)
    for qp in range(10):
        config_dict['qp'] = qp
        recon_array, bits = A3process.encode_complete(filepath, config_dict, table_dict)
        # reader.write_frame_array_to_file(recon_array, './files/CIF_y_recon' + str(qp) + '.yuv')
        # psnr = []
        # for num in range(frame):
        #     each_psnr = evaluation.calculate_psnr(origin_array[num], recon_array[num])
        #     psnr.append(each_psnr)
        # psnr_array.append(psnr)
        print("qp", qp)
        print(np.average(bits))

def generate_RD_plots():
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

    config = configparser.ConfigParser()
    config.read("./RD_plots_config.yaml")
    origin_array = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath), w, h, frame)
    psnr_array = []
    split_rate_array = []
    bits_array = []
    RD_plots_map = {}
    PSNR_arr_map = {}
    for section in config.sections():
        for qp in [1, 4, 7, 10]:
            print(section)
            nRefFrames = int(config[section]["nRefFrames"])
            VBSEnable = config[section]["VBSEnable"] == "True"
            FMEEnable = config[section]["FMEEnable"] == "True"
            FastME = config[section]["FastME"] == "True"
            n_block_w = (w - 1) // i + 1
            n_block_h = (h - 1) // i + 1
            lambda_coefficient = None
            if "lambda_coefficient" in config[section]:
                lambda_coefficient = float(config[section]['lambda_coefficient'])
            # print('case qp ' + str(qp) + ' lambda ' + str(lambda_coefficient))
            t = time.time()
            recon_array, bits = A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable,
                                                    lambda_coefficient, FMEEnable, FastME, frame)
            elapsed = time.time() - t
            bits_array.append(bits)
            reader.write_frame_array_to_file(recon_array, './files/foreman_cif_y_recon' + str(qp) + '_' + str(lambda_coefficient) + '.yuv')
            psnr = []
            for num in range(frame):
                each_psnr = evaluation.calculate_psnr(origin_array[num], recon_array[num])
                psnr.append(each_psnr)
            psnr_array.append(psnr)
            frame_count = 0
            print(section, qp, round(elapsed, 2))
        PSNR_arr_map[section] = psnr_array
        RD_plots_map[section] = bits_array
        bits_array = []
        psnr_array = []
    print(RD_plots_map)
    print(PSNR_arr_map)

def test_Nframe():
    filepath = './files/synthetic_y.yuv'
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
    nRefFrames = config_dict['nRefFrames']
    VBSEnable = config_dict['VBSEnable']
    # this is the coefficient to adjust lambda value
    lambda_coefficient = config_dict['lambda_coefficient']
    FMEEnable = config_dict['FMEEnable']
    FastME = config_dict['FastME']
    config = configparser.ConfigParser()
    config.read("./test_config.yaml")
    origin_array = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath), w, h, frame)
    psnr_array = []
    split_rate_array = []
    bits_array = []
    distortion_array = []
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1
    for section in config.sections():
        for nRefFrames in range(1, 5):
            lambda_coefficient = float(config[section]['lambda_coefficient'])
            print('case qp ' + str(qp) + ' lambda ' + str(lambda_coefficient))
            recon_array, bits = A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable,
                                                    lambda_coefficient, FMEEnable, FastME, frame)
            bits_array.append(bits)
            reader.write_frame_array_to_file(recon_array, './files/synthetic_y' + str(qp) + '_' + str(lambda_coefficient) + '.yuv')
            psnr = []
            distortion = []
            for num in range(frame):
                each_psnr = evaluation.calculate_psnr(origin_array[num], recon_array[num])
                each_distortion = evaluation.calculate_sad(origin_array[num], recon_array[num])
                psnr.append(each_psnr)
                distortion.append(each_distortion)
            psnr_array.append(psnr)
            distortion_array.append(distortion)
        print(psnr_array)
        print(bits_array)

    np.save('./files/psnr_arr.npy', psnr_array)
    np.save('./files/split_arr.npy', split_rate_array)
    np.save('./files/bits_arr.npy', np.array(bits_array))
    np.save('./files/distortion.npy', np.array(distortion_array))

if __name__ == '__main__':
    DELIVER_EX1()