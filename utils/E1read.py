import cv2 as cv
import numpy as np
import E1

yuv_to_rgb = [[1.164, 0, 1.596], [1.164, -0.392, -0.813], [1.164, 2.017, 0]]

def ready4m(filepath):
    video = cv.VideoCapture(filepath)

    if not video.isOpened():
        print('read fail')
        exit()

    return video


def release(video):
    video.release()


def read_raw(filepath):
    f = open(filepath, 'rb')
    return f


def read_raw_byte_array(filepath):
    f = open(filepath, 'rb')
    arr = bytes(f.read())
    f.close()
    return arr


def raw_yuv420_to_frame_array(byte_array, w, h):
    num_pixel = w * h
    num_bytes_per_frame = int(num_pixel * 1.5)
    num_bytes_per_uv = int(num_pixel / 4)
    num_frames = int(len(byte_array) / num_bytes_per_frame)
    frame_array = []
    for x in range(num_frames):
        print(x)
        frame_bytes = byte_array[x * num_bytes_per_frame: x * num_bytes_per_frame + num_bytes_per_frame]
        frame = np.zeros((h, w, 3))
        rgb = np.zeros((h, w, 3))
        w_2 = w // 2
        for i in range(num_pixel):
            frame[i // w][i % w][0] = frame_bytes[i]
        for i in range(num_bytes_per_uv):
            frame[(i // w_2) * 2][(i % w_2) * 2][1] = frame_bytes[num_pixel + i]
            frame[(i // w_2) * 2][(i % w_2) * 2 + 1][1] = frame_bytes[num_pixel + i]
            frame[(i // w_2) * 2 + 1][(i % w_2) * 2][1] = frame_bytes[num_pixel + i]
            frame[(i // w_2) * 2 + 1][(i % w_2) * 2 + 1][1] = frame_bytes[num_pixel + i]
            frame[(i // w_2) * 2][(i % w_2) * 2][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
            frame[(i // w_2) * 2][(i % w_2) * 2 + 1][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
            frame[(i // w_2) * 2 + 1][(i % w_2) * 2][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
            frame[(i // w_2) * 2 + 1][(i % w_2) * 2 + 1][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
        # for i in range(num_pixel):
        #     rgb[i // w][i % w] = np.dot(yuv_to_rgb, [frame[i // w][i % w][0] - 16, frame[i // w][i % w][1] - 128, frame[i // w][i % w][2] - 128])
        frame_array.append(frame)
    return frame_array


# 这个的位置对应是正确的
def read_one_frame(byte_array, w, h):
    num_pixel = w * h
    num_bytes_per_frame = int(num_pixel * 1.5)
    num_bytes_per_uv = int(num_pixel / 4)
    num_frames = int(len(byte_array) / num_bytes_per_frame)
    frame_bytes = byte_array[0: num_bytes_per_frame]
    frame = np.zeros((h, w, 3))
    w_2 = w // 2
    for i in range(num_pixel):
        frame[i // w][i % w][0] = frame_bytes[i]
    for i in range(num_bytes_per_uv):
        frame[(i // w_2) * 2][(i % w_2) * 2][1] = frame_bytes[num_pixel + i]
        frame[(i // w_2) * 2][(i % w_2) * 2 + 1][1] = frame_bytes[num_pixel + i]
        frame[(i // w_2) * 2 + 1][(i % w_2) * 2][1] = frame_bytes[num_pixel + i]
        frame[(i // w_2) * 2 + 1][(i % w_2) * 2 + 1][1] = frame_bytes[num_pixel + i]
        frame[(i // w_2) * 2][(i % w_2) * 2][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
        frame[(i // w_2) * 2][(i % w_2) * 2 + 1][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
        frame[(i // w_2) * 2 + 1][(i % w_2) * 2][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]
        frame[(i // w_2) * 2 + 1][(i % w_2) * 2 + 1][2] = frame_bytes[num_pixel + num_bytes_per_uv + i]

    return frame


if __name__ == '__main__':
    arr = read_raw_byte_array('./files/mother_daughter.yuv')
    # frame_array = raw_yuv420_to_frame_array(arr, 352, 288)
    # E1.play_raw_YUV(frame_array)
    frame = read_one_frame(arr, 352, 288)
    frame[:, :, 0] = frame[:, :, 0].clip(16, 235) - 16
    frame[:, :, 1:] = frame[:, :, 1:].clip(16, 240) -128