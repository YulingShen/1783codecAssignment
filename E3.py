from codec import E3process
from utils import reader
from matplotlib import pyplot as plt
import numpy as np
import time

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 3
    r = 4
    frames = 30
    t = time.time()
    MAE_arr, abs_res = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
    elapsed = time.time() - t
    print(i, r, elapsed)
    np.save("abs_res_r={}i={}".format(r, n, i), abs_res)
    # PSNR_arr = E3process.res_ME_decode(filepath, w, h, i)
    