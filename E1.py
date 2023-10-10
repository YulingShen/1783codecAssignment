import cv2 as cv
import numpy as np
from utils import E1read as reader


def play_y4m_yuv(video):
    while True:
        ret, frame = video.read()

        if not ret:
            break

        gray = frame[:, :, :1]
        gray2 = gray.reshape(len(frame), len(frame[0]))

        cv.imshow('Video', frame)
        cv.imshow('Gray', gray)
        cv.imshow('Gray2', gray2)
        cv.waitKey(1)
    cv.destroyAllWindows()

def play_raw_YUV(frame_array):
    for frame in frame_array:
        gray = frame[:, :, :1]
        cv.imshow('Video', frame)
        cv.imshow('Gray', gray)
        cv.waitKey(1)
    cv.destroyAllWindows()


# https://stackoverflow.com/questions/60704826/how-to-extract-frames-from-a-yuv-video-file-yuv420-using-python-and-opencv

if __name__ == '__main__':
    file = reader.ready4m('/Users/yulingshen/fall2023/1783/mother_daughter_cif.y4m')
    # file = reader.ready4m('/Users/yulingshen/fall2023/1783/mother_daughter.yuv')
    play_y4m_yuv(file)
    reader.release(file)
    cv.destroyAllWindows()


