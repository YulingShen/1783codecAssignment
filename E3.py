from codec import E3process
from utils import reader
from matplotlib import pyplot as plt
import numpy as np

def get_mae_plot_result(i, n, r, arr):
    plt.figure()
    arr1 = np.load("PSNR_r=4_i=8_n=1.png.npy")
    arr2 = np.load("PSNR_r=4_i=8_n=2.png.npy")
    arr3 = np.load("PSNR_r=4_i=8_n=3.png.npy")
    plt.plot(arr1, label="PSNR_n=1")
    plt.plot(arr2, label="PSNR_n=2")
    plt.plot(arr3, label="PSNR_n=3")
    plt.title("PSNR r=4 i=8 n=[1,2,3]")
    plt.xlabel('Frame')
    plt.ylabel('psnr (db)')
    plt.legend()
    plt.savefig('PSNR_r=4_i=8_n=[1,2,3].png')
    plt.close()

def get_psnr_plot_result(i, n, r, arr):
    plt.figure()
    arr1 = np.load("PSNR_r=4_i=8_n=1.png.npy")
    arr2 = np.load("PSNR_r=4_i=8_n=2.png.npy")
    arr3 = np.load("PSNR_r=4_i=8_n=3.png.npy")
    plt.plot(arr1, label="PSNR_n=1")
    plt.plot(arr2, label="PSNR_n=2")
    plt.plot(arr3, label="PSNR_n=3")
    plt.title("PSNR r=4 i=8 n=[1,2,3]")
    plt.xlabel('Frame')
    plt.ylabel('psnr (db)')
    plt.legend()
    plt.savefig('PSNR_r=4_i=8_n=[1,2,3].png')
    plt.close()

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 3
    r = 4
    frames = 10
    MAE_arr = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
    PSNR_arr = E3process.res_ME_decode(filepath, w, h, i)