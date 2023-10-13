from codec import E4process
from utils import reader

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 2
    r = 4
    qp = 2
    frames = None
    # E4process.encode_tran_quan(filepath, w, h, i, n, r, qp, frames)
    # reader.res_abs('./files/foreman_cif_y_res.yuv')
    E4process.encode_intra(filepath, w, h, i, n, frames)
