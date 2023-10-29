from codec import E4process
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
    E4process.encode_complete(filepath, w, h, i, n, r, qp, period, frame)
    E4process.decode_complete(filepath)
