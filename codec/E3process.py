from utils import reader
from codec import blocking
from codec.encoder import prediction_encode as pe
from codec.decoder import prediction_decode as pd
import numpy as np
from matplotlib import pyplot as plt

def res_no_ME_encode(filepath, w, h, i, n, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    prediction = np.full((h, w), 128, dtype=np.uint8)
    res_array = []
    abs_array = []
    recon_array = []
    for x in range(num_frames):
        print('encode fame: ' + str(x))
        res = prediction_encode.generate_residual(prediction, frame_block_array[x], w, h, n)
        res_array.append(res)
        abs_array.append(np.abs(res).astype(np.uint8))
        prediction = np.add(prediction, res).clip(0, 255).astype(np.uint8)
        recon_array.append(prediction)
    reader.write_frame_array_to_file(res_array, filepath[0:-4] + '_res.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[0:-4] + '_pred.yuv')

def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))

def res_ME_encode(filepath, w, h, i, n, r, num_frames=None):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, num_frames)
    if num_frames is None or len(frame_block_array) < num_frames:
        num_frames = len(frame_block_array)
    frame_block_array = frame_block_array[:num_frames]
    vectors = []
    res_array = []
    recon_array = []
    MAE_arr = []
    abs_array = []
    # print("We have", len(frame_block_array), " frames")
    decoder = pd.prediction_decoder()
    # padding
    prediction = np.full((h, w), 128, dtype=np.uint8)
    for x in range(num_frames):
        # print('encode fame: ' + str(x))
        res, vec, MAE = pe.generate_residual_ME(prediction, frame_block_array[x], w, h, n, r)
        res = blocking.deblock_frame(res, w, h)
        res_array.append(res)
        abs_array.append(np.abs(res).astype(np.uint8))
        prediction = decoder.decode_residual_ME(prediction, res, vec, w, h, i)
        recon_array.append(prediction)
        vectors.append(vec)
        MAE_arr.append(MAE)
    vectors = np.array(vectors, dtype=np.int8)
    reader.write_frame_array_to_file(res_array, filepath[:-4] + '_res_ME.yuv')
    reader.write_frame_array_to_file(recon_array, filepath[:-4] + '_pred_ME.yuv')
    np.save(filepath[:-4] + '_vec', vectors)
    return np.array(MAE_arr), np.array(abs_array)

def res_no_ME_decode(filepath, w, h):
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    raw_res = reader.read_raw_byte_array(filepath + '_res.yuv')
    frames_res = blocking.raw_to_frame(raw_res, w, h, np.int16)
    prediction = np.full((h, w), 128, dtype=np.uint8)
    video = []
    for x in range(len(frames_res)):
        print('decode frame: ' + str(x))
        prediction = np.add(prediction, frames_res[x]).clip(0, 255).astype(np.uint8)
        video.append(prediction)
    reader.write_frame_array_to_file(video, filepath + '_recon.yuv')


def res_ME_decode(filepath, w, h, i):
    y_only_bytes = reader.read_raw_byte_array(filepath)
    if filepath[-4:] == '.yuv':
        filepath = filepath[:-4]
    vecs = np.load(filepath + '_vec.npy')
    raw_res = reader.read_raw_byte_array(filepath + '_res_ME.yuv')
    frames_res = blocking.raw_to_frame(raw_res, w, h, np.int16)
    prediction = np.full((h, w), 128, dtype=np.uint8)
    video = []
    psnr_array = []
    frame_block_array = blocking.block_raw(y_only_bytes, w, h, i, len(vecs))
    frame_block_array = frame_block_array[:len(vecs)]
    decoder = pd.prediction_decoder()
    if len(vecs) != len(frames_res):
        print('different number of frames in the files, please check the configurations')
        return
    for x in range(len(vecs)):
        print('decode frame: ' + str(x))
        res = frames_res[x]
        vec = vecs[x]
        prediction = decoder.decode_residual_ME(prediction, res, vec, w, h, i)
        psnr_array.append(calculate_psnr(blocking.deblock_frame(frame_block_array[x], w, h), prediction))
        video.append(prediction)
    reader.write_frame_array_to_file(video, filepath + '_ME_recon.yuv')
    return np.array(psnr_array)