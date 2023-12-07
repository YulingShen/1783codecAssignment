from codec import A3process
from utils import reader
import time
import numpy as np

if __name__ == '__main__':
    filepath = './files/CIF_y.yuv'
    configpath = './config.yaml'
    config_dict = reader.load_config(configpath)
    tablepath = "./rate_table.yaml"
    table_dict = reader.load_RC_profile(tablepath)
    # config_dict["ParallelMode"] = 0
    # t = time.time()
    # recon_array_0, bits = A3process.encode_complete(filepath, config_dict, table_dict)
    # elapsed = time.time() - t
    # print(config_dict["ParallelMode"], "Execution Time:", elapsed)

    # config_dict["ParallelMode"] = 1
    # t = time.time()
    # recon_array_1, bits = A3process.encode_complete(filepath, config_dict, table_dict)
    # elapsed = time.time() - t
    # print(config_dict["ParallelMode"], "Execution Time:", elapsed)

    # config_dict["ParallelMode"] = 2
    # t = time.time()
    # recon_array_2, bits = A3process.encode_complete(filepath, config_dict, table_dict)
    # elapsed = time.time() - t
    # print(config_dict["ParallelMode"], "Execution Time:", elapsed)

    config_dict["ParallelMode"] = 3
    t = time.time()
    recon_array_3, bits = A3process.encode_complete(filepath, config_dict, table_dict)
    elapsed = time.time() - t
    print(config_dict["ParallelMode"], "Execution Time:", elapsed)

    print("Type 2 and Type 0 comparison", np.sum(abs(recon_array_2-recon_array0)))
    print("Type 3 and Type 0 comparison", np.sum(abs(recon_array_3-recon_array0)))
    # A3process.decode_complete(filepath)
