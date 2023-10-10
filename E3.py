from codec import process

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    w = 352
    h = 288
    i = 4
    n = 1
    r = 4
    frames = 30
    process.res_ME_encode(filepath, w, h, i, n, r, frames)
    process.res_ME_decode(filepath, w, h, i)
    process.res_no_ME_encode(filepath, w, h, i, n, frames)
    process.res_no_ME_decode(filepath, w, h)
