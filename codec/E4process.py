from utils import reader
from codec import blocking, quantization
from codec.encoder import prediction_encode, transform_encode, quantization_encode
from codec.decoder import prediction_decode, transform_decode, quantization_decode
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
        res = res.astype(np.int8)
        tran = transform_encode.transform_block(res)
        quan = quantization_encode.quantization_block(tran, q)
        quan_array.append(quan)
        # decode start
        dequan = quantization_decode.dequantization_block(quan, q)
        itran = transform_decode.inverse_transform_block(dequan).clip(-128, 127)
        res = blocking.deblock_frame(itran, w, h, True)
        # decode end
        res_array.append(res.astype(np.uint8))
        prediction = prediction_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        recon_array.append(prediction)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.int8)
    reader.write_frame_array_to_file(quan_array, filepath[:-4] + '_quan.yuv')
    reader.write_frame_array_to_file(res_array, filepath[:-4] + '_res.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[:-4] + '_pred.yuv')
    np.save(filepath[:-4] + '_vec', vectors)


def encode_intra(filepath, w, h, i, n, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    pred_array = []
    for x in range(num_frames):
        print('encode frame: ' + str(x))
        res, pred = prediction_encode.intra_residual(frame_block_array[x], n)
        pred = blocking.deblock_frame(pred, w, h)
        pred_array.append(pred)
    reader.write_frame_array_to_file(pred_array, filepath[:-4] + '_pred_intra.yuv')
