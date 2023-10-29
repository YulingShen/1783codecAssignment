from codec import E4process
from utils import reader
from matplotlib import pyplot as plt
import numpy as np
import configparser
import time

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 16
    n = 3
    r = 2
    qp = 3
    period = 4
    frames = 10

    # Average of PSNR

    config = configparser.ConfigParser()
    config.read("e4_config_comparison.yaml")
    bit_count_plot_arr = []
    RD_dict = {}
    p1_bits, p4_bits, p10_bits = [], [], []
    p1, p4, p10 = [], [], []
    for section in config.sections():
        i, qp, period = int(config[section]["i"]), int(config[section]["qp"]), int(config[section]["period"])
        t = time.time()
        bit_count, psnr = E4process.encode_complete(filepath, w, h, i, n, r, qp, period, frames)
        elapsed = time.time() - t
        print("i", i, "qp", qp, "period", period, "avg-psnr", np.average(psnr), "sum_bit_count", np.sum(bit_count), "time", elapsed)
        if period == 1:
            p1.append(np.average(psnr))
            p1_bits.append(np.sum(bit_count))
        elif period == 4:
            p4.append(np.average(psnr))
            p4_bits.append(np.sum(bit_count))
        else:
            p10.append(np.average(psnr))
            p10_bits.append(np.sum(bit_count))

        if qp == 3 or qp == 4:
            bit_count_plot_arr.append(bit_count)

        if len(bit_count_plot_arr) == 3:
            arr1, arr2, arr3 = bit_count_plot_arr
            plt.figure()
            arr_type = "Bit-Count Per Frame"
            plt.plot(arr1, label="period=1")
            plt.plot(arr2, label="period=4")
            plt.plot(arr3, label="period=10")
            plt.title("{} i={} qp={}".format(arr_type, i, qp))
            plt.xlabel('Frame')
            plt.ylabel('Bit-Count')
            plt.legend()
            plt.savefig("{}_i={}_qp={}.png".format(arr_type, i, qp))
            plt.close()
            bit_count_plot_arr = []

        if len(p1) == 4 and len(p4)==4 and len(p10)==4:
            arr1_x, arr2_x, arr3_x = p1_bits, p4_bits, p10_bits
            arr1, arr2, arr3 = p1, p4, p10
            plt.figure()
            arr_type = "RD Plot"
            plt.plot(arr1_x, arr1, label="period=1", marker='o', ls='-')
            plt.plot(arr2_x, arr2, label="period=4", marker='o', ls='-')
            plt.plot(arr3_x, arr3, label="period=10", marker='o', ls='-')
            plt.title("{} i={}".format(arr_type, i))
            plt.xlabel('Bit rate')
            plt.ylabel('PSNR (db)')
            plt.legend()
            plt.savefig("{}_i={}.png".format(arr_type, i))
            plt.close()
            p1_bits, p4_bits, p10_bits = [], [], []
            p1, p4, p10 = [], [], []

    # E4process.encode_tran_quan(filepath, w, h, i, n, r, qp, frames)
    # reader.res_abs('./files/foreman_cif_y_res.yuv')
    # E4process.encode_intra(filepath, w, h, i, n, qp, frames)
    # E4process.encode_intra_period(filepath, w, h, i, n, r, qp, period, frames)
    # E4process.decode_intra_period(filepath, w, h, i, qp, period)
    
    # E4process.decode_complete(filepath)
