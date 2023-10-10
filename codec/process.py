from utils import reader
from codec import blocking
from codec.encoder import residual_encode
from codec.decoder import residual_decode
import numpy as np


def res_no_ME_encode(filepath, w, h, i, n, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block(y_only_bytes, w, h, i)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction = np.full((h, w), 128, dtype=np.uint8)
    res_array = []
    recon_array = []
    for x in range(num_frames):
        print(x)
        res = residual_encode.generate_residual(prediction, frame_block_array[x], w, h, n)
        res_array.append(res)
        prediction = np.add(prediction, res)
        recon_array.append(prediction)
    reader.write_frame_array_to_file(res_array, filepath[0:-4] + '_res.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[0:-4] + '_pred.yuv')


def res_ME_encode(filepath, w, h, i, r, n, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block(y_only_bytes, w, h, i)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    vectors = []
    res_array = []
    recon_array = []
    prediction = np.full((h, w), 128, dtype=np.uint8)
    for x in range(num_frames):
        print(x)
        res, vec, MAE = residual_encode.generate_residual_ME(prediction, frame_block_array[x], w, h, n, r)
        res_array.append(res)
        prediction = residual_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        recon_array.append(prediction)
        vectors.append(vec)
    vectors = np.array(vectors, dtype=np.int8)
    reader.write_frame_array_to_file(res_array, filepath[:-4] + '_res_ME.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[:-4] + '_pred_ME.yuv')
    np.save(filepath[:-4] + '_vec', vectors)


def res_no_ME_decode(filepath, w, h):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    raw_res = reader.read_raw_byte_array(filepath + '_res.yuv')
    frames_res = blocking.raw_to_frame(raw_res, w, h)
    prediction = np.full((h, w), 128, dtype=np.uint8)
    video = []
    for x in range(len(frames_res)):
        print(x)
        prediction = np.add(prediction, frames_res[x])
        video.append(prediction)
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')


def res_ME_decode(filepath, w, h, i):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    vecs = np.load(filepath + '_vec.npy')
    raw_res = reader.read_raw_byte_array(filepath + '_res_ME.yuv')
    frames_res = blocking.raw_to_frame(raw_res, w, h)
    prediction = np.full((h, w), 128, dtype=np.uint8)
    video = []
    if len(vecs) != len(frames_res):
        print('different number of frames in the files, please check the configurations')
        return
    for x in range(len(vecs)):
        print(x)
        res = frames_res[x]
        vec = vecs[x]
        prediction = residual_decode.decode_residual_ME(prediction, res, vec, w, h, i)
        video.append(prediction)
    reader.write_frame_array_to_file(video, filepath + '_ME_recon.yuv')
