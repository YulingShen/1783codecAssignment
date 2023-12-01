from codec import A3process
from utils import reader

if __name__ == '__main__':
    filepath = './files/CIF_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    tablepath = "./rate_table.yaml"
    table_dict = reader.load_RC_profile(tablepath)
    # A3process.encode_complete(filepath, config_dict, table_dict)
    A3process.decode_complete(filepath)
