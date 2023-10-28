from codec import E3process
from utils import reader
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 4
    n = 3
    r = 4
    frames = 30
    for i in [4, 8, 16]:
        MAE_arr, abs_res = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
        np.save("abs_res_r={}i={}".format(r, i), abs_res)
    i = 8
    for r in [1, 4, 8]:
        MAE_arr, abs_res = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
        np.save("abs_res_r={}i={}".format(r, i), abs_res)
    # PSNR_arr = E3process.res_ME_decode(filepath, w, h, i)
    