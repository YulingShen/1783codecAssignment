import numpy as np

from utils import reader

if __name__ == '__main__':
    configpath = './config.yaml'
    cfg = reader.load_config(configpath)
    print(cfg)
