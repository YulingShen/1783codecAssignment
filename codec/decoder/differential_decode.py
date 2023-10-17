import numpy as np


def differential_decode(diff_array):
    diff_array = np.array(diff_array)
    vec = np.array(diff_array)
    if len(diff_array.shape) == 2:
        prev = np.array([0, 0])
        for x in range(diff_array.shape[0]):
            vec[x] = np.add(diff_array[x], prev)
            prev = vec[x]
    else:
        prev = 0
        for x in range(diff_array.shape[0]):
            vec[x] = diff_array[x] + prev
            prev = vec[x]
    return vec
