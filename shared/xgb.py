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
import NN
import xgboost
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


class XGBoost(NN.NeuralNetwork):
    def __init__(self, channel, gen, binary=True, write_filename='NN_output', show_graph=False):
        print(f'Loaded in {channel}, binary={binary}, gen={gen}')
        self.addons_config_reco = {'neutrino': {'load_alpha':False, 'termination':1000, 'imputer_mode':'remove', 'save_alpha':True,}, 'met': {}, 'ip': {}, 'sv': {}}
        self.addons_config_gen = {'neutrino': {'load_alpha':False, 'termination':1000, 'imputer_mode':'remove', 'save_alpha':True,}, 'met': {}, 'ip':{}, 'sv': {}}
        self.show_graph = show_graph
        self.channel = channel
        self.binary = binary
        self.write_filename = write_filename
        self.gen = gen
        self.save_dir = 'XGB_output'
        self.write_dir = 'XGB_output'
        self.model = None

    # def initializer(self, ...): This is inherited from NN

    def run(self, config_num, read=True, from_hdf=True,):
        """almost copy pasted from NN.py. The only changes are:
        arguments of run function
        arguments of train function
        print(f'Training with DEFAULT - xgboost model')
        self.history maybe I should remove that?"""
        print('checkpoint 1')
        if not self.gen:
            df = self.initialize(self.addons_config_reco, read=read, from_hdf=from_hdf)
        else:
            df = self.initialize(self.addons_config_gen, read=read, from_hdf=from_hdf)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if self.model is None:
            print(f'Training with DEFAULT - xgboost model')
        model = self.train(X_train, X_test, y_train, y_test)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, None)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, None, w_a, w_b)
        print('Writing...')
        # self.write(self.gen, auc, self.addons_config_reco) # !!! this crashed the code so I changed it 26 Feb
        self.write(auc, self.addons_config_reco)

    def runTuning(self, config_num, tuning_mode='hyperopt'):
        """
        INCOMPLETE AND UNTESTED
        """
        if not self.gen:
            df = self.initialize(self.addons_config_reco, read=read, from_hdf=from_hdf)
        else:
            df = self.initialize(self.addons_config_gen, read=read, from_hdf=from_hdf)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tuning XGB on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        tuner = Tuner(mode=tuning_mode)
        if tuning_mode == 'hyperopt':
            self.model, best_params, param_grid = tuner.tuneXGB(X_train, y_train, X_test, y_test)

            self.learning_rate = float(best_params['learning_rate'])
            self.max_depth = int(best_params['max_depth'])
            self.min_child_weight = int(best_params['min_child_weight'])
            self.gamma = float(best_params['gamma'])
            self.subsample = float(best_params['subsample'])
            self.colsample_bytree = float(best_params['colsample_bytree'])
            self.reg_alpha = float(best_params['reg_alpha'])
            self.model_str = 'hyperopt_XGB_model'
        else:
            raise ValueError('Tuning mode not understood')
        model = self.train(X_train, X_test, y_train, y_test, stopping_rounds=100, save=False, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, None)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, None, w_a, w_b)
        if not self.gen:
            file = f'{self.write_dir}/tuning_xgb_reco_{self.channel}.txt'
        else:
            file = f'{self.write_dir}/tuning_xgb_gen_{self.channel}.txt'
        #self.model.save(f'{self.write_dir}/tuning_xgb_model_{self.channel}.h5')
        with open(file, 'a+') as f:
            print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            # message = f'{time_str},{auc},{self.config_num},{self.layers},{self.batch_norm},{self.dropout},{self.epochs},{self.batchsize},{tuning_mode},{grid_best_score},{param_grid}\n'
            message = f'{time_str},{auc},{self.config_num},{self.learning_rate},{self.max_depth},{self.min_child_weight},{self.gamma},{self.subsample},{self.colsample_bytree},{self.reg_alpha},{tuning_mode},{param_grid}\n'
            print(f"Message: {message}")
            f.write(message)
        model_save_str = f'./saved_models/{self.channel}/xgb_model_{config_num}'
        self.model.save_model(model_save_str)

    def createConfigStr(self):
        """almost copy pasted. Differences:
        erased some variables from config_str"""
        self.model_str = 'XGBoost'
        if self.binary:
            config_str = f'config{self.config_num}_{self.model_str}_binary'
        else:
            config_str = f'config{self.config_num}_{self.model_str}'
        return config_str

    def write(self, auc, addons_config):
        """almost copy pasted. Differences:
        actual_epochs deleted
        f.write(...) erased epochs, batch size, etc."""
        if not addons_config:
            addons = []
        else:
            addons = addons_config.keys()
        addons_loaded = "None"
        if addons:
            addons_loaded = '_'+'_'.join(addons)
        if not self.gen:
            file = f'{self.write_dir}/{self.write_filename}_reco_{self.channel}.txt'
        else:
            file = f'{self.write_dir}/{self.write_filename}_gen_{self.channel}.txt'    
        with open(file, 'a+') as f:
            print(f'Writing to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            f.write(f'{time_str},{auc},{self.config_num},{self.binary},{self.model_str},{addons_loaded}\n')
        print('Finish writing')
        f.close()

    def train(self, X_train, X_test, y_train, y_test, stopping_rounds=100, save=False, verbose=1):
        if self.model is None:
            self.model = self.xgbModel()
        self.model.fit(X_train, y_train,
            early_stopping_rounds=stopping_rounds, # stops the training if doesn't improve after 200 iterations
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric = "auc", # can use others
            verbose=verbose)
        if save:
            self.model.save_model(f'./saved_models/{self.save_dir}/xgboost.json')
        return self.model

    def xgbModel(self):
        xgb_params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.02,
            "silent": 1,
            "n_estimators": 100,
            "subsample": 0.9,
            "seed": config.seed_value,
        }
        xgb_clf = xgboost.XGBClassifier(**xgb_params)
        return xgb_clf


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', default='rho_rho', choices=['rho_rho', 'rho_a1', 'a1_a1'], help='which channel to load to')
    parser.add_argument('config_num', type=float, help='config num to run on')
    parser.add_argument('-g', '--gen', action='store_true', default=False, help='if load gen data')
    parser.add_argument('-b', '--binary', action='store_false', default=True, help='if learn binary labels')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='if tuning is run')
    parser.add_argument('-tm', '--tuning_mode', help='choose tuning mode to tune on', default='hyperopt')
    parser.add_argument('-r', '--read', action='store_false', default=True, help='if read NN input')
    parser.add_argument('-hdf', '--from_hdf', action='store_false', default=True, help='if read .root file from HDF5')
    parser.add_argument('-a', '--addons', nargs='*', default=None, help='load addons')
    parser.add_argument('-s', '--show_graph', action='store_true', default=False, help='if show graphs')
    parser.add_argument('-la', '--load_alpha', action='store_false', default=True, help='if load alpha')
    parser.add_argument('-ter', '--termination', type=int, default=1000, help='termination number for alpha')
    parser.add_argument('-imp', '--imputer_mode', default='remove', choices=['pass', 'flag', 'bayesian_ridge', 'decision_tree', 'extra_trees', 'kn_reg', 'knn', 'mean', 'remove'], help='imputation mode for neutrino information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    if not os.path.exists('C:\\Kristof'):  # then we are on Stanley's computer
        # print(tf.test.is_built_with_cuda(), tf.config.list_physical_devices('GPU'))
        # exit()
        # use command line parser - comment out if not needed
        args = parser()
        channel = args.channel
        config_num = args.config_num / 10
        gen = args.gen
        binary = args.binary
        tuning = args.tuning
        tuning_mode = args.tuning_mode
        read = args.read
        from_hdf = args.from_hdf
        addons = args.addons
        show_graph = args.show_graph
        load_alpha = args.load_alpha
        termination = args.termination
        imputer_mode = args.imputer_mode

        XGB = XGBoost(channel=channel, gen=gen, binary=binary, write_filename='XGB_output', show_graph=show_graph)
        if not tuning:
            print('checkpoint 0')
            print('checkpoint 0.5')
            XGB.run(config_num, read=read, from_hdf=from_hdf)
        else:
            XGB.runTuning(config_num, tuning_mode=tuning_mode)
    else:
        pass