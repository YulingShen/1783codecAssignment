import numpy as np

from utils import reader
from codec import blocking
from codec.encoder import residual_encode
from codec.decoder import residual_decode


def non_motion():
    y_only_bytes = reader.read_raw_byte_array('./files/foreman_cif_y.yuv')
    frame_block_array = blocking.block(y_only_bytes, 352, 288, 4)
    # num_frames = len(frame_block_array)
    num_frames = 10
    frame_block_array = frame_block_array[:10]
    prediction = np.full((288, 352), 128, dtype=np.uint8)
    res_array = []
    recon_array = []
    for x in range(num_frames):
        print(x)
        res = residual_encode.generate_residual(prediction, frame_block_array[x], 352, 288, 3)
        res_array.append(res)
        prediction = np.add(prediction, res)
        recon_array.append(prediction)
    reader.write_frame_array_to_file(res_array, './files/foreman_cif_res.yuv')
    reader.write_frame_array_to_file(recon_array, './files/foreman_cif_recon.yuv')


def motion_compensation():
    y_only_bytes = reader.read_raw_byte_array('./files/foreman_cif_y.yuv')
    frame_block_array = blocking.block(y_only_bytes, 352, 288, 4)
    # num_frames = len(frame_block_array)
    num_frames = 10
    frame_block_array = frame_block_array[:10]
    vectors = []
    res_array = []
    recon_array = []
    prediction = np.full((288, 352), 128, dtype=np.uint8)
    for x in range(num_frames):
        print(x)
        res, vec, MAE = residual_encode.generate_residual_ME(prediction, frame_block_array[x], 352, 288, 3, 4)
        res_array.append(res)
        prediction = residual_decode.decode_residual_ME(prediction, res, vec, 352, 288, 4)
        recon_array.append(prediction)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.int8)
    reader.write_frame_array_to_file(res_array, './files/foreman_cif_res_ME.yuv')
    reader.write_frame_array_to_file(recon_array, './files/foreman_cif_recon_ME.yuv')
    # reader.save_vector_ME(vectors, './files/foreman_cif_vecb')
    np.save('./files/test', vectors)

if __name__ == '__main__':
    # non_motion()
    motion_compensation()
