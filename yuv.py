import numpy as np
import cv2

def read_YUV420(frame_width, frame_height, file_path):

    with open(file_path, 'rb') as file:

        frame_size = frame_width * frame_height + 2 * (frame_width // 2) * (frame_height // 2)


        raw = file.read(frame_size)
        while raw:

            Y = np.frombuffer(raw[:frame_width * frame_height], dtype=np.uint8).reshape((frame_height, frame_width))
            U = np.frombuffer(raw[frame_width * frame_height:frame_width * frame_height + (frame_width // 2) * (frame_height // 2)], dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))
            V = np.frombuffer(raw[frame_width * frame_height + (frame_width // 2) * (frame_height // 2):], dtype=np.uint8).reshape((frame_height // 2, frame_width // 2))

            U = cv2.resize(U, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
            V = cv2.resize(V, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

            YUV = cv2.merge([Y, U, V])
            BGR = cv2.cvtColor(YUV, cv2.COLOR_YUV2BGR)
            cv2.imshow('Frame', BGR)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            raw = file.read(frame_size)

    cv2.destroyAllWindows()

# Usage
file_path = './files/vbs_frame0.yuv'
frame_width = 1920
frame_height = 1080
read_YUV420(frame_width, frame_height, file_path)
