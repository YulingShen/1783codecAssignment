import numpy as np


def differential_encode(vector_array):
    vector_array = np.array(vector_array)
    diff = np.array(vector_array)
    if len(vector_array.shape) == 2:
        prev = np.array([0, 0])
        for x in range(vector_array.shape[0]):
            diff[x] = np.subtract(vector_array[x], prev)
            prev = vector_array[x]
    else:
        prev = 0
        for x in range(vector_array.shape[0]):
            diff[x] = vector_array[x] - prev
            prev = vector_array[x]
    return diff
