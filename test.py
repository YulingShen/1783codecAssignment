import numpy as np

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    # print(a)
    b = [a, a]
    print([b[i][0] for i in range(2)])