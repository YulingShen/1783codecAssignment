import configparser

import numpy as np
import cv2

from codec import A2process
from codec.decoder import entropy_decode, differential_decode
from utils import reader


def draw_arrow(image, start_point, vector, block_size):
    end_point = (int(start_point[0] + vector[0]), int(start_point[1] + vector[1]))
    arrowed_image = cv2.arrowedLine(image, start_point, end_point, color=(0, 0, 0), thickness=1, tipLength=0.3)
    return arrowed_image


def apply_color_to_block(image_shape, center, block_size, k):
    color_map = {
        1: (128, 0, 0),
        2: (0, 0, 128),
        3: (128, 128, 0),
    }

    color = np.array(color_map.get(k, (0, 128, 0)))
    colored_layer = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
    y_start = center[1]
    x_start = center[0]

    colored_layer[y_start:y_start + block_size, x_start:x_start + block_size, :] = color

    return colored_layer


def apply_mode_to_block(image_shape, center, block_size, k):
    color_map = {
        1: (128, 0, 0),

    }
    color = np.array(color_map.get(k, (0, 128, 0)))

    colored_layer = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)

    y_start = center[1]
    x_start = center[0]
    colored_layer[y_start:y_start + block_size, x_start:x_start + block_size, :] = color

    return colored_layer


def draw_border(frame, row, col, block_size, is_split):
    y_start = row * block_size
    x_start = col * block_size
    border_thickness = 1

    if is_split:
        half_size = block_size // 2
        frame[y_start:y_start + border_thickness, x_start:x_start + block_size] = 0
        frame[y_start + half_size - border_thickness:y_start + half_size, x_start:x_start + block_size] = 0
        frame[y_start + block_size - border_thickness:y_start + block_size, x_start:x_start + block_size] = 0
        frame[y_start:y_start + block_size, x_start:x_start + border_thickness] = 0
        frame[y_start:y_start + block_size, x_start + half_size - border_thickness:x_start + half_size] = 0
        frame[y_start:y_start + block_size, x_start + block_size - border_thickness:x_start + block_size] = 0
    else:
        frame[y_start:y_start + border_thickness, x_start:x_start + block_size] = 0
        frame[y_start + block_size - border_thickness:y_start + block_size, x_start:x_start + block_size] = 0
        frame[y_start:y_start + block_size, x_start:x_start + border_thickness] = 0
        frame[y_start:y_start + block_size, x_start + block_size - border_thickness:x_start + block_size] = 0

    return frame


def overlay_mode_blocks_on_gray_image(gray_image, split_decision, mode, block_size, alpha=0.4):
    n_block_h, n_block_w = gray_image.shape[0] // block_size, gray_image.shape[1] // block_size
    colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

    mode_index = 0
    for row in range(n_block_h):
        for col in range(n_block_w):
            block_split = split_decision[row * n_block_w + col]
            if block_split == 1:
                # center = (col * block_size + block_size // 2, row * block_size + block_size // 2)
                center = (col * block_size, row * block_size)
                mode_value = mode[mode_index]
                colored_block = apply_mode_to_block(gray_image.shape, center, block_size, mode_value)
                colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                mode_index += 4
                # 处理分割的四个小块
                # sub_block_size = block_size // 2
                # sub_blocks = [(row*block_size, col*block_size), #纵坐标，横坐标
                #               (row*block_size, col*block_size + sub_block_size), 
                #               (row*block_size + sub_block_size, col*block_size), 
                #               (row*block_size + sub_block_size, col*block_size + sub_block_size)]   
                # mode_value = mode[mode_index] 
                # for sub_block in sub_blocks:
                #     center = (sub_block[1] + sub_block_size // 2, sub_block[0] + sub_block_size // 2)
                #     # mode_value = mode[mode_index]
                #     colored_block = apply_mode_to_block(gray_image.shape, center, block_size, mode_value)
                #     colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                #     mode_index += 1                       
            else:
                # 处理未分割的大块
                center = (col * block_size, row * block_size)
                mode_value = mode[mode_index]
                colored_block = apply_mode_to_block(gray_image.shape, center, block_size, mode_value)
                colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                mode_index += 1

    gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    final_image = cv2.addWeighted(gray_image_3ch, 1, colored_image, alpha, 0)
    return final_image


def overlay_color_blocks_on_gray_image(gray_image, split_decision, vec, block_size, alpha=0.4):
    """
    Overlays colored blocks on a gray image based on the values in vec.
    """
    n_block_h, n_block_w = gray_image.shape[0] // block_size, gray_image.shape[1] // block_size
    colored_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)

    vec_index = 0
    for row in range(n_block_h):
        for col in range(n_block_w):
            block_split = split_decision[row * n_block_w + col]
            if block_split == 1:
                center = (col * block_size, row * block_size)
                vec_value = vec[vec_index]
                colored_block = apply_color_to_block(gray_image.shape, center, block_size, vec_value[2])
                colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                vec_index += 4
                # 处理分割的四个小块
                # sub_block_size = block_size // 2
                # sub_blocks = [(row*block_size, col*block_size), #纵坐标，横坐标
                #               (row*block_size, col*block_size + sub_block_size), 
                #               (row*block_size + sub_block_size, col*block_size), 
                #               (row*block_size + sub_block_size, col*block_size + sub_block_size)]    
                # vec_value = vec[vec_index]
                # for sub_block in sub_blocks:
                #     center = (sub_block[1] + sub_block_size // 2, sub_block[0] + sub_block_size // 2)
                #     # vec_value = vec[vec_index]
                #     colored_block = apply_color_to_block(gray_image.shape, center, block_size, vec_value[2])
                #     colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                #     vec_index += 1                       
            else:
                # 处理未分割的大块
                center = (col * block_size, row * block_size)
                vec_value = vec[vec_index]
                colored_block = apply_color_to_block(gray_image.shape, center, block_size, vec_value[2])
                colored_image = cv2.addWeighted(colored_image, 1, colored_block, alpha, 0)
                vec_index += 1

    gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    final_image = cv2.addWeighted(gray_image_3ch, 1, colored_image, alpha, 0)
    return final_image


def process_frame_for_arrows(current_frame, split_decision, vec, block_size):
    n_block_h, n_block_w = current_frame.shape[0] // block_size, current_frame.shape[1] // block_size
    arrow_frame = np.copy(current_frame)
    vec_index = 0

    for row in range(n_block_h):
        for col in range(n_block_w):
            block_split = split_decision[row * n_block_w + col]

            if block_split == 1:
                # 处理分割的四个小块
                sub_block_size = block_size // 2
                sub_blocks = [(row * block_size, col * block_size),  # 纵坐标，横坐标
                              (row * block_size, col * block_size + sub_block_size),
                              (row * block_size + sub_block_size, col * block_size),
                              (row * block_size + sub_block_size, col * block_size + sub_block_size)]

                for sub_block in sub_blocks:
                    center = (sub_block[1] + sub_block_size // 2, sub_block[0] + sub_block_size // 2)
                    vec_value = vec[vec_index]
                    arrow_frame = draw_arrow(arrow_frame, center, vec_value[:2], sub_block_size)
                    # color_frame = apply_color_to_block(color_frame, center, sub_block_size, vec_value[2])
                    vec_index += 1
            else:
                # 处理未分割的大块
                center = (col * block_size + block_size // 2, row * block_size + block_size // 2)
                vec_value = vec[vec_index]
                arrow_frame = draw_arrow(arrow_frame, center, vec_value[:2], block_size)
                # color_frame = apply_color_to_block(color_frame, center, block_size, vec_value[2])
                vec_index += 1

    return arrow_frame


if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'

    with open(filepath[:-4] + '_res', 'r') as res_file:
        res_code = res_file.read()
    with open(filepath[:-4] + '_diff', 'r') as vec_file:
        setting = vec_file.readline()
        vec_code = vec_file.read()

    w, h, i, qp, period, VBSEnable, FMEEnable = entropy_decode.decode_setting(setting)

    origin_array = reader.y_only_byte_frame_array(reader.read_raw_byte_array(filepath), w, h)
    splits = []
    vecs = []
    n_block_w = (w - 1) // i + 1
    n_block_h = (h - 1) // i + 1

    frame_count = 0
    if not VBSEnable:
        while len(vec_code) > 0:
            if frame_count % period == 0:
                diff_array, vec_code = entropy_decode.decode_vec_one_frame(vec_code, n_block_h * n_block_w, False)
                mode_array = differential_decode.differential_decode(diff_array)
                vecs.append(mode_array)
            else:
                diff_array, vec_code = entropy_decode.decode_vec_one_frame(vec_code, n_block_h * n_block_w, True)
                vec_array = differential_decode.differential_decode(diff_array)
                vecs.append(vec_array)
            frame_count += 1
    else:
        while len(vec_code) > 0:
            split_diff, vec_code = entropy_decode.decode_split_one_frame(vec_code, n_block_h * n_block_w)
            split_array = differential_decode.differential_decode(split_diff)
            len_array = n_block_h * n_block_w + 3 * np.sum(split_array)
            if frame_count % period == 0:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, False)
                mode_array = differential_decode.differential_decode(vec_diff)
                vecs.append(mode_array)
            else:
                vec_diff, vec_code = entropy_decode.decode_vec_one_frame(vec_code, len_array, True)
                vec_array = differential_decode.differential_decode(vec_diff)
                vecs.append(vec_array)
            frame_count += 1
            splits.append(split_array)

    for num in range(frame_count):
        current_frame = origin_array[num]
        split_array = np.zeros(n_block_h * n_block_w)
        if VBSEnable:
            split_array = splits[num]
            border_frame = np.copy(current_frame)
            for row in range(n_block_h):
                for col in range(n_block_w):
                    block_split = split_array[row * n_block_w + col]
                    border_frame = draw_border(border_frame, row, col, i, block_split == 1)
            reader.write_frame_array_to_file(border_frame, './files/visualize/border_frame_' + str(num) + '.yuv')
        if num % period == 0:
            mode_frame = overlay_mode_blocks_on_gray_image(current_frame, split_array, vecs[num], i)
            reader.write_frame_array_to_file(mode_frame, './files/visualize/mode_frame_' + str(num) + '.yuv')
        else:
            arrow_frame = process_frame_for_arrows(current_frame, split_array, vecs[num], i)
            color_frame = overlay_color_blocks_on_gray_image(current_frame, split_array, vecs[num], i)
            reader.write_frame_array_to_file(arrow_frame, './files/visualize/arrow_frame_' + str(num) + '.yuv')
            reader.write_frame_array_to_file(color_frame, './files/visualize/ref_frame_' + str(num) + '.yuv')
