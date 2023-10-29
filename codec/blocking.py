import numpy as np


def raw_to_frame(y_only_arr, w, h, dtype=np.uint8):
    num_size = np.dtype(dtype).itemsize
    num_pixel = w * h
    num_bytes = num_size * num_pixel
    num_frames = int(len(y_only_arr) / num_bytes)
    frames = []
    for x in range(num_frames):
        print(x)
        frame = np.zeros((h, w), dtype=dtype)
        frame_bytes = y_only_arr[x * num_bytes: x * num_bytes + num_bytes]
        for n in range(num_pixel):
            frame[n // w][n % w] = int.from_bytes(frame_bytes[n * num_size: n * num_size + num_size], 'little')
        frames.append(frame)
    return frames


def raw_to_block(y_only_arr, w, h, i, dtype=np.uint8):
    w_count = (w - 1) // i + 1
    h_count = (h - 1) // i + 1
    num_size = np.dtype(dtype).itemsize
    num_pixel = w * h
    num_bytes = num_size * num_pixel
    num_frames = int(len(y_only_arr) / num_bytes)
    frames = []
    for x in range(num_frames):
        print(x)
        frame = np.full((h_count, w_count, i, i), 128, dtype=dtype)
        frame_bytes = y_only_arr[x * num_bytes: x * num_bytes + num_bytes]
        for n in range(num_pixel):
            block_count = n // (i * i)
            n_h = block_count // w_count
            n_w = block_count % w_count
            frame[n_h][n_w][(n // i) % i][n % i] = int.from_bytes(
                frame_bytes[n * num_size: n * num_size + num_size], 'little')
        frames.append(frame)
    return frames


def block_raw(y_only_arr, w, h, i, num_frames=None, dtype=np.uint8):
    w_count = (w - 1) // i + 1
    h_count = (h - 1) // i + 1
    num_size = np.dtype(dtype).itemsize
    num_pixel = w * h
    num_bytes = num_size * num_pixel
    if num_frames is None or int(len(y_only_arr) / num_bytes) < num_frames:
        num_frames = int(len(y_only_arr) / num_bytes)
    frames = []
    for x in range(num_frames):
        #　print(x)
        frame = np.full((h_count, w_count, i, i), 128, dtype=dtype)
        frame_bytes = y_only_arr[x * num_bytes: x * num_bytes + num_bytes]
        for n in range(num_pixel):
            n_h = n // w
            n_w = n % w
            frame[n_h // i][n_w // i][n_h % i][n_w % i] = int.from_bytes(frame_bytes[n * num_size: n * num_size + num_size], 'little')
        frames.append(frame)
    return frames


def block_frame(frame, i):
    h = len(frame)
    w = len(frame[0])
    w_count = (w - 1) // i + 1
    h_count = (h - 1) // i + 1
    result = np.full((h_count, w_count, i, i), 128, dtype=frame.dtype)
    for x in range(h):
        for y in range(w):
            result[x // i][y // i][x % i][y % i] = frame[x][y]
    return result


def deblock(frame_block_arr, w=None, h=None):
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


def deblock_frame(frame_block, w=None, h=None):
    block_size = len(frame_block[0][0])
    h_count = len(frame_block)
    w_count = len(frame_block[0])
    if w is None:
        w = w_count * block_size
    if h is None:
        h = h_count * block_size
    result = np.zeros((h, w), dtype=frame_block.dtype)
    for i in range(h):
        for j in range(w):
            result[i][j] = frame_block[i // block_size][j // block_size][i % block_size][j % block_size]
    return result
