from codec import E3process
from utils import reader
from matplotlib import pyplot as plt
import numpy as np
import configparser

def get_R_plot(arr_type):
    plt.figure()
    arr1 = np.load("{}_r=1n=3i=8.npy".format(arr_type))
    arr2 = np.load("{}_r=4n=3i=8.npy".format(arr_type))
    arr3 = np.load("{}_r=8n=3i=8.npy".format(arr_type))
    plt.plot(arr1, label="{}_r=1".format(arr_type))
    plt.plot(arr2, label="{}_r=4".format(arr_type))
    plt.plot(arr3, label="{}_r=8".format(arr_type))
    plt.title("{} r=[1,4,8] n=3 i=8".format(arr_type))
    plt.xlabel('Frame')
    if arr_type == "PSNR":
        plt.ylabel('psnr (db)')
    else:
        plt.ylabel('mae')
    plt.legend()
    plt.savefig("{}_r=[1,4,8]n=3i=8.png".format(arr_type))
    plt.close()

def get_N_plot(arr_type):
    plt.figure()
    arr1 = np.load("{}_r=4n=1i=8.npy".format(arr_type))
    arr2 = np.load("{}_r=4n=2i=8.npy".format(arr_type))
    arr3 = np.load("{}_r=4n=3i=8.npy".format(arr_type))
    plt.plot(arr1, label="{}_n=1".format(arr_type))
    plt.plot(arr2, label="{}_n=2".format(arr_type))
    plt.plot(arr3, label="{}_n=3".format(arr_type))
    plt.title("{} r=4 n=[1,2,3] i=8".format(arr_type))
    plt.xlabel('Frame')
    if arr_type == "PSNR":
        plt.ylabel('psnr (db)')
    else:
        plt.ylabel('mae')
    plt.legend()
    plt.savefig("{}_r=4n=[1,2,3]i=8.png".format(arr_type))
    plt.close()

def get_I_plot(arr_type):
    plt.figure()
    arr1 = np.load("{}_r=4n=3i=4.npy".format(arr_type))
    arr2 = np.load("{}_r=4n=3i=8.npy".format(arr_type))
    arr3 = np.load("{}_r=4n=3i=16.npy".format(arr_type))
    plt.plot(arr1, label="{}_i=4".format(arr_type))
    plt.plot(arr2, label="{}_i=8".format(arr_type))
    plt.plot(arr3, label="{}_i=16".format(arr_type))
    plt.title("{} r=4 n=3 i=[4,8,16]".format(arr_type))
    plt.xlabel('Frame')
    if arr_type == "PSNR":
        plt.ylabel('psnr (db)')
    else:
        plt.ylabel('mae')
    plt.legend()
    plt.savefig("{}_r=4n=3i=[4,8,16].png".format(arr_type))
    plt.close()

if __name__ == '__main__':
    filepath = './files/mad900_y.yuv'
    w = 176
    h = 144
    frames = 20
    r, n, i = 4,3,8
    config = configparser.ConfigParser()
    config.read("config_comparison.yaml")

    config_dict = {}
    for section in config.sections():
        r, n, i = int(config[section]["r"]), int(config[section]["n"]), int(config[section]["i"])
        MAE_arr, _ = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
        PSNR_arr = E3process.res_ME_decode(filepath, w, h, i) 
        np.save("PSNR_r={}n={}i={}".format(r, n, i), PSNR_arr)
        np.save("MAE_r={}n={}i={}".format(r, n, i), MAE_arr)
    

    get_R_plot("PSNR")
    get_R_plot("MAE")

    get_N_plot("PSNR")
    get_N_plot("MAE")

    get_I_plot("PSNR")
    get_I_plot("MAE")
