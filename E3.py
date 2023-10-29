from codec import E3process
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
    frame = config_dict['frame']
    E3process.res_ME_encode(filepath, w, h, i, n, r, frame)
    E3process.res_ME_decode(filepath, w, h, i)
    E3process.res_no_ME_encode(filepath, w, h, i, n, frame)
    E3process.res_no_ME_decode(filepath, w, h)
    # generate the prediction used for motion estimation
    E3process.ME_prediction(filepath, w, h, i)
    # generate absolute residual magnitude
    reader.res_abs(filepath[:-4] + '_res_ME.yuv')
    reader.res_abs(filepath[:-4] + '_res.yuv')
