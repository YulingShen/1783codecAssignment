from codec import evaluation
from utils import reader

if __name__ == '__main__':
    origin = reader.y_only_byte_frame_array(reader.read_raw_byte_array('./files/foreman_cif_y.yuv'), 352, 288)
    recon = reader.y_only_byte_frame_array(reader.read_raw_byte_array('./files/foreman_cif_y_recon.yuv'), 352, 288)
    psnr_arr = []
    ssd_arr = []
    for i in range(10):
        ssd = evaluation.calculate_ssd(origin[i], recon[i])
        psnr = evaluation.calculate_psnr(origin[i], recon[i])
        ssd_arr.append(ssd)
        psnr_arr.append(psnr)
    print(psnr_arr)
    print(ssd_arr)