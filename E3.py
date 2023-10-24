from codec import E3process
from utils import reader
from matplotlib import pyplot as plt

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 3
    r = 4
    frames = 30
    for n in range(1,5):
        try:
            MAE_arr = E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
            PSNR_arr = E3process.res_ME_decode(filepath, w, h, i)
            plt.figure()
            plt.plot(MAE_arr)
            plt.title("MAE r={} i={} n={}".format(r, i, n))
            plt.xlabel('Frame')
            plt.ylabel('mae')
            plt.savefig('MAE_r={}_i={}_n={}.png'.format(r, i, n))
            plt.close()

            plt.figure()
            plt.plot(PSNR_arr)
            plt.title("PSNR r={} i={} n={}".format(r, i, n))
            plt.xlabel('Frame')
            plt.ylabel('psnr')
            plt.savefig('PSNR_r={}_i={}_n={}.png'.format(r, i, n))
            plt.close()
        except:
            print("Failed", i)
    """
    A per-frame PSNR graph measured as PSNR between original and reconstructed frames
    as well as a per-frame average MAE graph calculated during the MV selection process.
    Show a set of graphs for varying 𝑖 with fixed 𝑟=4 and 𝑛=3, another for varying 𝑟 with fixed 𝑖=8 and 𝑛=3,
    and another for varying 𝑛 with fixed 𝑖=8 and 𝑟=4. You can use any sequences you want,
    but the first 10 frames of Foreman CIF (352x288) are a requirement, 
    plus at least one other sequence of different dimensions (number of frames at your discretion).
    """
    
    # E3process.res_no_ME_encode(filepath, w, h, i, n, frames)
    # E3process.res_no_ME_decode(filepath, w, h)
