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
    p1_bits, p4_bits, p10_bits = [], [], []
    p1, p4, p10 = [], [], []
    execution_time_dict = {1:[], 4:[], 10:[]}
    for section in config.sections():
        i, qp, period = int(config[section]["i"]), int(config[section]["qp"]), int(config[section]["period"])
        t = time.time()
        bit_count, psnr = E4process.encode_complete(filepath, w, h, i, n, r, qp, period, frames)
        elapsed = time.time() - t
        execution_time_dict[period].append(round(elapsed, 2))

        # np.save("PSNR_qp={}p={}i={}".format(qp, period, i), psnr)
        # np.save("Bit_qp={}p={}i={}".format(qp, period, i), bit_count)
        print("i", i, "qp", qp, "period", period, "avg-psnr", np.average(psnr), "sum_bit_count", np.sum(bit_count), "time", elapsed)
        # if period == 1:
        #     p1.append(np.average(psnr))
        #     p1_bits.append(np.sum(bit_count))
        # elif period == 4:
        #     p4.append(np.average(psnr))
        #     p4_bits.append(np.sum(bit_count))
        # else:
        #     p10.append(np.average(psnr))
        #     p10_bits.append(np.sum(bit_count))

        # if qp == 3 or qp == 4:
        #     bit_count_plot_arr.append(bit_count)

        # if len(bit_count_plot_arr) == 3:
        #     arr1, arr2, arr3 = bit_count_plot_arr
        #     plt.figure()
        #     arr_type = "Bit-Count Per Frame"
        #     plt.plot(arr1, label="period=1")
        #     plt.plot(arr2, label="period=4")
        #     plt.plot(arr3, label="period=10")
        #     plt.title("{} i={} qp={}".format(arr_type, i, qp))
        #     plt.xlabel('Frame')
        #     plt.ylabel('Bit-Count')
        #     plt.legend()
        #     plt.savefig("{}_i={}_qp={}.png".format(arr_type, i, qp))
        #     plt.close()
        #     bit_count_plot_arr = []

        # if len(p1) == 4 and len(p4)==4 and len(p10)==4:
        #     arr1_x, arr2_x, arr3_x = p1_bits, p4_bits, p10_bits
        #     arr1, arr2, arr3 = p1, p4, p10
        #     z1 = np.polyfit(arr1_x, arr1, 5)
        #     z2 = np.polyfit(arr2_x, arr2, 5)
        #     z3 = np.polyfit(arr3_x, arr3, 5)
        #     f1, f2, f3 = np.poly1d(z1), np.poly1d(z2), np.poly1d(z3)
        #     plt.figure()
        #     arr_type = "RD Plot"
        #     y1, y2, y3 = f1(arr1_x), f2(arr2_x), f3(arr3_x)
        #     plt.plot(arr1_x, arr1, label="period=1", marker='o')
        #     plt.plot(arr2_x, arr2, label="period=4", marker='o')
        #     plt.plot(arr3_x, arr3, label="period=10", marker='o')
        #     plt.plot(arr1_x, y1, label="period=1")
        #     plt.plot(arr2_x, y2, label="period=4")
        #     plt.plot(arr3_x, y3, label="period=10")
        #     plt.title("{} i={}".format(arr_type, i))
        #     plt.xlabel('Bit rate')
        #     plt.ylabel('PSNR (db)')
        #     plt.legend()
        #     plt.savefig("{}_i={}.png".format(arr_type, i))
        #     plt.close()
        #     p1_bits, p4_bits, p10_bits = [], [], []
        #     p1, p4, p10 = [], [], []

        # if len(execution_time_dict[10]) == 4:
        #     if i == 8:
        #         species = (0, 3, 6, 9)
        #     else:
        #         species = (1, 4, 7, 10)
        #     width = 0.25  # the width of the bars
        #     multiplier = 0
        #     fig, ax = plt.subplots(layout='constrained')
        #     x = np.arange(len(species))
        #     for attribute, measurement in execution_time_dict.items():
        #         offset = width * multiplier
        #         rects = ax.bar(x + offset, measurement, width, label="period={}".format(attribute))
        #         ax.bar_label(rects, padding=3)
        #         multiplier += 1

        #     # Add some text for labels, title and custom x-axis tick labels, etc.
        #     ax.set_ylabel('Execution time (s)')
        #     ax.set_title('Execution time of different settings')
        #     ax.set_xticks(x + width, species)
        #     ax.legend(loc='upper left', ncols=3)
        #     ax.set_ylim(0, 12)
        #     # plt.show()
        #     ax.set_title('Total execution time of 10 frames in different settings with i={}'.format(i))
        #     plt.savefig("i{}_time_plot.png".format(i))
        #     plt.close()
        #     execution_time_dict = {1:[], 4:[], 10:[]}

    # E4process.encode_tran_quan(filepath, w, h, i, n, r, qp, frames)
    # reader.res_abs('./files/foreman_cif_y_res.yuv')
    # E4process.encode_intra(filepath, w, h, i, n, qp, frames)
    # E4process.encode_intra_period(filepath, w, h, i, n, r, qp, period, frames)
    # E4process.decode_intra_period(filepath, w, h, i, qp, period)
    
    # E4process.decode_complete(filepath)
