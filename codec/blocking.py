import numpy as np


def block(y_only_arr, w, h, i):
    w_count = (w - 1) // i + 1
    h_count = (h - 1) // i + 1
    num_pixel = w * h
    num_frames = int(len(y_only_arr) / num_pixel)
    frames = []
    for x in range(num_frames):
        print(x)
        frame = np.full((h_count, w_count, i, i), 128, dtype=np.uint8)
        frame_bytes = y_only_arr[x * num_pixel: x * num_pixel + num_pixel]
        for n in range(num_pixel):
            n_h = n // w
            n_w = n % w
            frame[n_h // i][n_w // i][n_h % i][n_w % i] = frame_bytes[n]
        frames.append(frame)
    return frames


def block_frame(frame, i):
    h = len(frame)
    w = len(frame[0])
    w_count = (w - 1) // i + 1
    h_count = (h - 1) // i + 1
    result = np.full((h_count, w_count, i, i), 128, dtype=np.uint8)
    for x in range(h):
        for y in range(w):
            result[x // i][y // i][x % i][y % i] = frame[x][y]
    return result


def deblock(frame_block_arr, w = None, h = None):
    block_size = len(frame_block_arr[0][0][0])
    h_count = len(frame_block_arr[0])
    w_count = len(frame_block_arr[0][0])
    if w is None:
        w = w_count * block_size
    if h is None:
        h = h_count * block_size
    frames = []
    num_frames = len(frame_block_arr)
    for x in range(num_frames):
        print(x)
        blocked_frame = frame_block_arr[x]
        frame = np.zeros((h_count * block_size, w_count * block_size), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                frame[i][j] = blocked_frame[i // block_size][j // block_size][i % block_size][j % block_size]
        frames.append(frame)
    return frames


def deblock_frame(frame_block, w = None, h = None):
    block_size = len(frame_block[0][0])
    h_count = len(frame_block)
    w_count = len(frame_block[0])
    if w is None:
        w = w_count * block_size
    if h is None:
        h = h_count * block_size
    result = np.zeros((h, w), dtype=np.uint8)
    # for i in range(h_count):
    #     for j in range(w_count):
    #         for x in range(block_size):
    #             for y in range(block_size):
    #                 result[i * block_size + x][j * block_size + y] = frame_block[i][j][x][y]
    for i in range(h):
        for j in range(w):
            result[i][j] = frame_block[i // block_size][j // block_size][i % block_size][j % block_size]

    return result


if __name__ == '__main__':
    import reader

    arr = reader.read_raw_byte_array('/Users/yulingshen/fall2023/1783/mother_daughter_y.yuv')
    frame_blocks = block(arr, 352, 288, 2)
    frames = deblock(frame_blocks)
    reader.write_frame_array_to_file(frames, '/Users/yulingshen/fall2023/1783/mother_daughter_recon.yuv')
