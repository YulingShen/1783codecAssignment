import numpy as np

from codec import A3process
from codec.encoder import entropy_encode
from utils import reader

if __name__ == "__main__":
    print(entropy_encode.exp_golomb(1))