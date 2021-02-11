from evaluator import Evaluator
from data_loader import DataLoader
from config_loader import ConfigLoader
from config_checker import ConfigChecker
from tuner import Tuner
import os
import tensorflow as tf
import random
import numpy as np
import datetime
import config
import argparse
from utils import TensorBoardExtended
seed_value = config.seed_value
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)

config_tf = tf.compat.v1.ConfigProto()
config_tf.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config_tf)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

print("GPU list: ", tf.config.list_physical_devices('GPU'))


class XGBoost:


    def __init__(self):
        pass



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', default='rho_rho', choices=['rho_rho', 'rho_a1', 'a1_a1'], help='which channel to load to')
    parser.add_argument('config_num', type=float, help='config num to run on')
    parser.add_argument('-g', '--gen', action='store_true', default=False, help='if load gen data')
    parser.add_argument('-b', '--binary', action='store_false', default=True, help='if learn binary labels')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='if tuning is run')
    parser.add_argument('-tm', '--tuning_mode', help='choose tuning mode to tune on', default='random_sk')
    parser.add_argument('-r', '--read', action='store_false', default=True, help='if read NN input')
    parser.add_argument('-hdf', '--from_hdf', action='store_false', default=True, help='if read .root file from HDF5')
    parser.add_argument('-a', '--addons', nargs='*', default=None, help='load addons')
    parser.add_argument('-s', '--show_graph', action='store_true', default=False, help='if show graphs')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='epochs to train on')
    parser.add_argument('-bs', '--batch_size', type=int, default=10000, help='batch size')
    parser.add_argument('-la', '--load_alpha', action='store_false', default=True, help='if load alpha')
    parser.add_argument('-ter', '--termination', type=int, default=1000, help='termination number for alpha')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if not os.path.exists('C:\\Kristof'):  # then we are on Stanley's computer
        # print(tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))
        # exit()
        # use command line parser - comment out if not needed
        args = parser()
        channel = args.channel
        config_num = args.config_num 
        gen = args.gen
        binary = args.binary
        tuning = args.tuning
        tuning_mode = args.tuning_mode
        read = args.read
        from_hdf = args.from_hdf
        addons = args.addons
        show_graph = args.show_graph
        epochs = args.epochs
        batch_size = args.batch_size
        load_alpha = args.load_alpha
        termination = args.termination
    else:
        pass