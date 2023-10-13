import numpy as np


def generate_q(i, qp):
    if qp < 0 or qp > np.log2(i) + 7:
        return None
    q = np.full((i, i), 2 ** qp)
    for x in range(i):
        y = i - x - 1
        q[x][y] = q[x][y] * 2
    for x in range(1, i):
        for y in range(i - x, i):
            q[x][y] = q[x][y] * 4
    return q
