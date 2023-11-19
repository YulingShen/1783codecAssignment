import numpy as np


def calculate_psnr(img1, img2, max_value=255):
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1) - np.array(img2)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


# followings are for RDO
def get_lambda(qp, lambda_coefficient):
    return lambda_coefficient * 2 ** ((qp - 12) / 3)


# we would like to use SSD(sum of square difference) or SAD(sum of absolute difference)
def calculate_ssd(arr1, arr2):
    return np.sum((np.array(arr1) - np.array(arr2)) ** 2)

# SAD(sum of absolute difference)
def calculate_sad(arr1, arr2):
    return np.sum(np.abs(np.array(arr1) - np.array(arr2)))

def calculate_rdo(ssd, bit_count, lambda_val):
    return ssd + lambda_val * bit_count
