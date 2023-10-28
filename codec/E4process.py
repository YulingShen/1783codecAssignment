from utils import reader
from codec import blocking, quantization
from codec.encoder import prediction_encode, transform_encode, quantization_encode, differential_encode, entropy_encode
from codec.decoder import prediction_decode, transform_decode, quantization_decode, differential_decode, entropy_decode
import numpy as np


def encode_tran_quan(filepath, w, h, i, n, r, qp, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    vectors = []
    quan_array = []
    res_array = []
    recon_array = []
    prediction = np.full((h, w), 128, dtype=np.uint8)
    q = quantization.generate_q(i, qp)
    for x in range(num_frames):
        print('encode fame: ' + str(x))
        res, vec, MAE = prediction_encode.generate_residual_ME(prediction, frame_block_array[x], w, h, n, r)
        tran = transform_encode.transform_frame(res)
        quan = quantization_encode.quantization_frame(tran, q)
        # decode start
        dequan = quantization_decode.dequantization_frame(quan, q)
        itran = transform_decode.inverse_transform_frame(dequan).clip(-128, 127)
        res = blocking.deblock_frame(itran, w, h)
        # decode end
        res_array.append(res)
        prediction = prediction_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        recon_array.append(prediction)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.int8)
    reader.write_frame_array_to_file(quan_array, filepath[:-4] + '_quan.yuv')
    reader.write_frame_array_to_file(res_array, filepath[:-4] + '_res.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[:-4] + '_pred.yuv')
    np.save(filepath[:-4] + '_vec', vectors)


def encode_intra(filepath, w, h, i, n, qp, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    q = quantization.generate_q(i, qp)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    pred_array = []
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        res, pred, modes = prediction_encode.intra_residual(frame_block_array[x], n, q)
        pred = blocking.deblock_frame(pred, w, h)
        pred_array.append(pred)
    reader.write_frame_array_to_file(pred_array, filepath[:-4] + '_pred_intra.yuv')


def encode_intra_period(filepath, w, h, i, n, r, qp, period, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    vectors = []
    quan_array = []
    res_array = []
    recon_array = []
    prediction = np.full((h, w), 128, dtype=np.uint8)
    q = quantization.generate_q(i, qp)
    for x in range(num_frames):
        print('encode fame: ' + str(x))
        if x % period == 0:
            res, prediction, vec, quan_frame = prediction_encode.intra_residual(frame_block_array[x], n, q)
            quan_array.append(quan_frame)
            res = blocking.deblock_frame(res, w, h)
            prediction = blocking.deblock_frame(prediction, w, h)
        else:
            res, vec, MAE = prediction_encode.generate_residual_ME(prediction, frame_block_array[x], w, h, n, r)
            tran = transform_encode.transform_frame(res)
            quan = quantization_encode.quantization_frame(tran, q)
            quan_array.append(quan)
            # decode start
            dequan = quantization_decode.dequantization_frame(quan, q)
            itran = transform_decode.inverse_transform_frame(dequan).clip(-128, 127)
            res = blocking.deblock_frame(itran, w, h)
        # decode end
        res_array.append(res)
        if x % period != 0:
            prediction = prediction_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        recon_array.append(prediction)
        vectors.append(vec)
    reader.write_frame_array_to_file(quan_array, filepath[:-4] + '_quan.yuv')
    reader.write_frame_array_to_file(res_array, filepath[:-4] + '_res.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[:-4] + '_pred.yuv')
    np.save(filepath[:-4] + '_vec', vectors)


def decode_intra_period(filepath, w, h, i, qp, period):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    vecs = np.load(filepath + '_vec.npy', allow_pickle=True)
    raw_quan = reader.read_raw_byte_array(filepath + '_quan.yuv')
    # raw_to_block arranges bytes in the order of saved per block
    frames_quan = blocking.raw_to_block(raw_quan, w, h, i, dtype=np.int16)
    prediction = np.full((h, w), 128, dtype=np.uint8)
    video = []
    q = quantization.generate_q(i, qp)
    for x in range(len(vecs)):
        print('decode frame: ' + str(x))
        quan = frames_quan[x]
        dequan = quantization_decode.dequantization_frame(quan, q)
        itran = transform_decode.inverse_transform_frame(dequan)
        res = blocking.deblock_frame(itran, w, h)
        vec = vecs[x]
        if x % period == 0:
            prediction = prediction_decode.intra_decode(res, vec, w, h, i)
        else:
            prediction = prediction_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        video.append(prediction)
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')


# final process controller for E4 encoding
def encode_complete(filepath, w, h, i, n, r, qp, period, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    # config is written to the first line of diff file
    line, bits = entropy_encode.entropy_encode_setting(w, h, i, qp, period)
    diff_file.write(line)
    diff_file.write("\n")
    bit_sum = 0
    bit_count = 0
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction = np.full((h, w), 128, dtype=np.uint8)
    q = quantization.generate_q(i, qp)
    bit_count_arr=[]
    for x in range(num_frames):
        print('encode fame: ' + str(x))
        if x % period == 0:
            res, prediction, vec, quan_frame = prediction_encode.intra_residual(frame_block_array[x], n, q)
            code, bit_count = entropy_encode.entropy_encode_quan_frame_block(quan_frame)
            residual_file.write(code)
            bit_sum += bit_count
            code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
            diff_file.write(code)
            bit_sum += bit_count
            res = blocking.deblock_frame(res, w, h)
            prediction = blocking.deblock_frame(prediction, w, h)
        else:
            res, vec, MAE = prediction_encode.generate_residual_ME(prediction, frame_block_array[x], w, h, n, r)
            tran = transform_encode.transform_frame(res)
            quan = quantization_encode.quantization_frame(tran, q)
            code, bit_count = entropy_encode.entropy_encode_quan_frame_block(quan)
            residual_file.write(code)
            bit_sum += bit_count
            code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
            diff_file.write(code)
            bit_sum += bit_count
            # decode start
            dequan = quantization_decode.dequantization_frame(quan, q)
            itran = transform_decode.inverse_transform_frame(dequan).clip(-128, 127)
            res = blocking.deblock_frame(itran, w, h)
        # decode end
        bit_count_arr.append(bit_count)
        if x % period != 0:
            prediction = prediction_decode.decode_residual_ME(prediction, res, vec, w, h, i)
    residual_file.close()
    diff_file.close()
    return bit_count
    # print(bit_sum)


# file process controller of E4 decoding
def decode_complete(filepath):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, i, qp, period = entropy_decode.decode_setting(setting)
    q = quantization.generate_q(i, qp)
    video = []
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1
    frame_count = 0
    while len(res_code) > 0 and len(vec_code) > 0:
        print("decode frame: " + str(frame_count))
        quan, res_code = entropy_decode.decode_quan_one_frame(res_code, n_block_w, n_block_h, i)
        dequan = quantization_decode.dequantization_frame(quan, q)
        itran = transform_decode.inverse_transform_frame(dequan)
        res = blocking.deblock_frame(itran, w, h)
        if frame_count % period == 0:
            diff_array, vec_code = entropy_decode.decode_vec_one_frame(vec_code, n_block_h * n_block_w, False)
            mode_array = differential_decode.differential_decode(diff_array)
            recon = prediction_decode.intra_decode(res, mode_array, w, h, i)
        else:
            diff_array, vec_code = entropy_decode.decode_vec_one_frame(vec_code, n_block_h * n_block_w, True)
            vec_array = differential_decode.differential_decode(diff_array)
            recon = prediction_decode.decode_residual_ME(recon, res, vec_array, w, h, i)
        video.append(recon)
        frame_count += 1
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')
