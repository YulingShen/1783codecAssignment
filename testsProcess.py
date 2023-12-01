from codec import blocking, quantization, evaluation
from codec.decoder import prediction_decode
from codec.encoder import entropy_encode, prediction_encode, differential_encode
from utils import reader

import numpy as np

if __name__ == '__main__':
    filepath = './files/QCIF_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    w = config_dict['w']
    h = config_dict['h']
    i = config_dict['i']
    n = config_dict['n']
    r = config_dict['r']
    qp = config_dict['qp']
    period = config_dict['period']
    VBSEnable = config_dict['VBSEnable']
    nRefFrames = config_dict['nRefFrames']
    VBSEnable = config_dict['VBSEnable']
    # this is the coefficient to adjust lambda value
    lambda_coefficient = config_dict['lambda_coefficient']
    FMEEnable = config_dict['FMEEnable']
    FastME = config_dict['FastME']

    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i)
    # files to write
    # config is written to the first line of diff file
    num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    # prediction = np.full((h, w), 128, dtype=np.uint8)
    for qp in range(0, 12):
        print("run qp: " + str(qp))
        q = quantization.generate_q(i, qp)
        prediction_array = []
        bit_count_arr = []

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
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                bit_sum += bit_count
                prediction = blocking.deblock_frame(prediction, w, h)
                prediction_array = [prediction]
            else:
                block_itran, vec, split, res_code = prediction_encode.generate_residual_ME_VBS(prediction_array,
                                                                                               frame_block_array[x], w,
                                                                                               h, n, r, lambda_val, q,
                                                                                               q_split, FMEEnable,
                                                                                               FastME)
                bit_sum += len(res_code)
                # write split indicators first, then vectors
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(split))
                bit_sum += bit_count
                code, bit_count = entropy_encode.entropy_encode_vec(differential_encode.differential_encode(vec))
                bit_sum += bit_count
                res = blocking.deblock_frame(block_itran)
                prediction = prediction_decode.decode_residual_ME_VBS(prediction_array, res, vec, split, w, h, i,
                                                                      FMEEnable)
                prediction_array.insert(0, prediction)
                if len(prediction_array) >= nRefFrames:
                    prediction_array = prediction_array[:nRefFrames]
            bit_count_arr.append(bit_sum)
        np.asarray(bit_count_arr).tofile("./files/bit_count_qp_" + str(qp) + ".csv", sep=',')
