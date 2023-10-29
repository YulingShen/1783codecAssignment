from codec import E3process
from utils import reader

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 3
    r = 4
    frames = 30
    E3process.res_ME_encode(filepath, w, h, i, n, r, frames)
    E3process.res_ME_decode(filepath, w, h, i)
    E3process.res_no_ME_encode(filepath, w, h, i, n, frames)
    E3process.res_no_ME_decode(filepath, w, h)
    E3process.ME_prediction(filepath, w, h, i)
    reader.res_abs(filepath[:-4] + '_res_ME.yuv')
    reader.res_abs(filepath[:-4] + '_res.yuv')
