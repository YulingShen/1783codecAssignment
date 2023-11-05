from codec import A2process
from utils import reader

if __name__ == '__main__':
    filepath = './files/foreman_cif_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    w = config_dict['w']
    h = config_dict['h']
    i = config_dict['i']
    n = config_dict['n']
    r = config_dict['r']
    qp = config_dict['qp']
    period = config_dict['period']
    frame = config_dict['frame']
    VBSEnable = config_dict['VBSEnable']
    nRefFrames = config_dict['nRefFrames']
    FMEEnable = config_dict['FMEEnable']
    A2process.encode_complete(filepath, w, h, i, n, r, qp, period, nRefFrames, FMEEnable, frame)
    A2process.decode_complete(filepath)
