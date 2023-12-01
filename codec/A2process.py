from utils import reader
from codec import blocking, quantization, evaluation
from codec.encoder import prediction_encode, transform_encode, quantization_encode, differential_encode, entropy_encode
from codec.decoder import prediction_decode, transform_decode, quantization_decode, differential_decode, entropy_decode
import numpy as np


def encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, VBSEnable, lambda_coefficient, FMEEnable, FastME, RCFlag,
                    num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    # files to write
    residual_file = open(filepath[:-4] + '_res', 'w')
    diff_file = open(filepath[:-4] + '_diff', 'w')
    # config is written to the first line of diff file
    line, bits = entropy_encode.entropy_encode_setting(w, h, i, qp, period, VBSEnable, FMEEnable, RCFlag)
    diff_file.write(line)
    diff_file.write("\n")
    bit_count_arr = []
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    # prediction = np.full((h, w), 128, dtype=np.uint8)
    q = quantization.generate_q(i, qp)
    prediction_array = []
    if not VBSEnable:
        for x in range(num_frames):
            print('encode frame: ' + str(x))
            bit_sum = 0
            if x % period == 0:
                res, prediction, vec, quan_frame = prediction_encode.intra_residual(frame_block_array[x], n, q)
                code, bit_count = entropy_encode.entropy_encode_quan_frame_block(quan_frame)
                residual_file.write(code)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_sum += bit_count
                prediction = blocking.deblock_frame(prediction, w, h)
                prediction_array = [prediction]
            else:
                res, vec = prediction_encode.generate_residual_ME(prediction_array, frame_block_array[x], w, h, n,
                                                                  r, FMEEnable, FastME)
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
                prediction = prediction_decode.decode_residual_ME(prediction_array, res, vec, w, h, i, FMEEnable)
                prediction_array.insert(0, prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
            bit_count_arr.append(bit_sum)
    else:
        lambda_val = evaluation.get_lambda(qp, lambda_coefficient)
        if qp > 0:
            q_split = quantization.generate_q(int(i / 2), qp - 1)
        else:
            q_split = quantization.generate_q(int(i / 2), qp)
        for x in range(num_frames):
            print('encode frame: ' + str(x))
            bit_sum = 0
            # here the transform and quantization are done within the prediction part
            # as it needs these information to decide if sub blocks takes less r-d cost
            if x % period == 0:
                prediction, vec, split, res_code = prediction_encode.intra_residual_VBS(frame_block_array[x], n,
                                                                                        lambda_val, q, q_split)
                residual_file.write(res_code)
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                diff_file.write(code)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_sum += bit_count
                prediction = blocking.deblock_frame(prediction, w, h)
                prediction_array = [prediction]
            else:
                block_itran, vec, split, res_code = prediction_encode.generate_residual_ME_VBS(prediction_array,
                                                                                               frame_block_array[x], w,
                                                                                               h, n, r, lambda_val, q,
                                                                                               q_split, FMEEnable,
                                                                                               FastME)
                residual_file.write(res_code)
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                diff_file.write(code)
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                diff_file.write(code)
                bit_sum += bit_count
                res = blocking.deblock_frame(block_itran)
                prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec, split, w, h, i,
                                                                      FMEEnable)
                prediction_array.insert(0, prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
            bit_count_arr.append(bit_sum)
    residual_file.close()
    diff_file.close()


def decode_complete(filepath):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    with open(filepath + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()
    w, h, i, qp, period, VBSEnable, FMEEnable, RCFlag = entropy_decode.decode_setting(setting)
    q = quantization.generate_q(i, qp)
    video = []
    recon_array = []
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1
    frame_count = 0
    if not VBSEnable:
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
                recon_array = [recon]
            else:
                diff_array, vec_code = entropy_decode.decode_vec_one_frame(vec_code, n_block_h * n_block_w, True)
                vec_array = differential_decode.differential_decode(diff_array)
                recon = prediction_decode.decode_residual_ME(recon_array, res, vec_array, w, h, i, FMEEnable)
                recon_array.insert(0, recon)
                # here use the upper limit of nRefFrame for easier coding purpose
                if len(recon_array) == 4:
                    recon_array = recon_array[:4]
            video.append(recon)
            frame_count += 1
    else:
        if qp > 0:
            q_split = quantization.generate_q(int(i / 2), qp - 1)
        else:
            q_split = quantization.generate_q(int(i / 2), qp)
        while len(res_code) > 0 and len(vec_code) > 0:
            print("decode frame: " + str(frame_count))
            # get array for split indication
            split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_block_h * n_block_w)
            split_array = differential_decode.differential_decode(split_diff)
            len_array = n_block_h * n_block_w + 3 * np.sum(split_array)
            # get quantization of the frame
            quan, res_code = entropy_decode.decode_quan_one_frame_VBS(res_code, n_block_w, n_block_h, i, split_array)
            dequan = quantization_decode.dequantization_frame_VBS(quan, q, q_split, split_array)
            itran = transform_decode.inverse_transform_frame_VBS(dequan, split_array)
            res = blocking.deblock_frame(itran, w, h)
            if frame_count % period == 0:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, False)
                mode_array = differential_decode.differential_decode(vec_diff)
                recon = prediction_decode.intra_decode_VBS(res, mode_array, split_array, w, h, i)
                recon_array = [recon]
            else:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, True)
                vec_array = differential_decode.differential_decode(vec_diff)
                recon = prediction_decode.decode_residual_ME_VBS(recon_array, res, vec_array, split_array, w, h, i,
                                                                 FMEEnable)
                recon_array.insert(0, recon)
                # here use the upper limit of nRefFrame for easier coding purpose
                if len(recon_array) == 4:
                    recon_array = recon_array[:4]
            video.append(recon)
            frame_count += 1
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')
