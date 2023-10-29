from codec import E4process
from utils import reader

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 8
    n = 3
    r = 2
    qp = 6
    period = 4
    frames = 10
    E4process.encode_complete(filepath, w, h, i, n, r, qp, period, frames)
    E4process.decode_complete(filepath)
