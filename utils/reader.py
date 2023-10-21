import numpy as np
import sys


def read_raw_byte_array(filepath):
    f = open(filepath, 'rb')
    arr = bytes(f.read())
    f.close()
    return arr


def y_only_byte_frame_array(y_only_arr, w, h):
    num_pixel = w * h
    num_frames = int(len(y_only_arr) / num_pixel)
    frames = []
    for x in range(num_frames):
        print("read frame: " + str(x))
        frame = np.zeros((h, w), dtype=np.uint8)
        frame_bytes = y_only_arr[x * num_pixel: x * num_pixel + num_pixel]
        for n in range(num_pixel):
            n_h = n // w
            n_w = n % w
            frame[n_h][n_w] = frame_bytes[n]
        frames.append(frame)
    return frames


def byte_array_y_only(byte_array, w, h, format):
    ratio = 1
    if format == '420':
        ratio = 1.5
    elif format == '444':
        ratio = 3
    elif format == '422':
        ratio = 2
    num_pixel = w * h
    num_bytes_per_frame = int(num_pixel * ratio)
    num_frames = int(len(byte_array) / num_bytes_per_frame)
    y_only_arr = bytes()
    for x in range(num_frames):
        y_only_arr = y_only_arr.__add__(byte_array[x * num_bytes_per_frame: x * num_bytes_per_frame + num_pixel])
    return y_only_arr


def write_byte_array_to_file(byte_array, filepath):
    f = open(filepath, 'wb')
    f.write(byte_array)
    f.close()


def write_frame_array_to_file(frame_array, filepath):
    f = open(filepath, 'wb')
    for frame in frame_array:
    #     for i in range(len(frame)):
    #         f.write(bytes(frame[i]))
        f.write(bytes(frame))
    f.close()


def write_frame_block_array_to_file(frame_block_array, filepath):
    f = open(filepath, 'wb')
    for frame_block in frame_block_array:
        for i in range(len(frame_block)):
            for j in range(len(frame_block[0])):
                for k in range(len(frame_block[0][0])):
                    f.write(bytes(frame_block[i][j][k]))
    f.close()

def res_abs(filepath):
    residual = np.fromfile(filepath, dtype=np.int16)
    residual = np.abs(residual).astype(np.uint8)
    f = open(filepath[:-4] + '_abs.yuv', 'wb')
    f.write(bytes(residual))
    f.close()


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('reader.py file_path w h format, format being 444, 422, 420')
    file_path = sys.argv[1]
    w = int(sys.argv[2])
    h = int(sys.argv[3])
    format = sys.argv[4]
    if file_path[-4:] != '.yuv':
        print('please use raw yuv file')
    arr = read_raw_byte_array(file_path)
    y_only = byte_array_y_only(arr, w, h, format)
    save_file = file_path[0:-4] + '_y.yuv'
    write_byte_array_to_file(y_only, save_file)
