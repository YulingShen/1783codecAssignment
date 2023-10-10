from codec import process

if __name__ == '__main__':
    process.res_ME_encode('./files/foreman_cif_y.yuv', 352, 288, 4, 4, 3, 10)
    process.res_ME_decode('./files/foreman_cif_y.yuv', 352, 288, 4)
    process.res_no_ME_encode('./files/foreman_cif_y.yuv', 352, 288, 4, 3, 10)
    process.res_no_ME_decode('./files/foreman_cif_y.yuv', 352, 288)
