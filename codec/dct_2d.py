from scipy.fftpack import dct, idct


def dct_2d(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idct_2d(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')
