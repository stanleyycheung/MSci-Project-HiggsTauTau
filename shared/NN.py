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
import kerastuner as kt
import config
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import argparse
seed_value = config.seed_value
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)


class NeuralNetwork:
    """
    Features
    - Run a single config_num
    - Run multiple config_nums
    - Will automatically write and save results of model - see save_dir and write_dir for more information
    Notes:
    - Supports Tensorboard
    """

    def __init__(self,  channel, gen, binary=True, write_filename='NN_output', show_graph=False):
        self.show_graph = show_graph
        self.channel = channel
        self.binary = binary
        self.write_filename = write_filename
        self.gen = gen
        self.save_dir = 'NN_output'
        self.write_dir = 'NN_output'
        self.model = None

    def run(self, config_num, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10, addons_config={}):
        if not self.gen:
            df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        else:
            df = self.initializeGen(read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if self.model is None:
            print(f'Training with DEFAULT - kristof_model')
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        if not self.gen:
            self.write(auc, self.history, addons_config)
        else:
            self.writeGen(auc, self.history)

    def runWithNeutrino(self, config_num, load_alpha=False, termination=1000, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10):
        """
        Does not support gen
        """
        if self.gen:
            print("runWithNeutrino does not support gen")
            raise SystemExit
        addons_config={'neutrino': {'load_alpha':load_alpha, 'termination':termination}, 'met':{}}
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num, mode=1)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        self.write(auc, self.history, addons_config)

    def runMultiple(self, configs, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10, addons_config={}):
        if not self.gen:
            df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        else:
            df = self.initializeGen(read=read, from_pickle=from_pickle)
        for config_num in configs:
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            X_train, X_test, y_train, y_test = self.configure(df, config_num)
            model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience)
            if self.binary:
                auc = self.evaluateBinary(model, X_test, y_test, self.history)
            else:
                w_a = df.w_a
                w_b = df.w_b
                auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
            if not self.gen:
                self.write(auc, self.history, addons_config)
            else:
                self.writeGen(auc, self.history)

    def runTuning(self, config_num, tuning_mode='random_sk', addons_config={}, read=True, from_pickle=True):
        if not self.gen:
            df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        else:
            df = self.initializeGen(read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tuning on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        tuner = Tuner(mode=tuning_mode)
        if tuning_mode in {'random_sk', 'grid_search_cv'}:
            model_grid, grid_result, param_grid = tuner.tune(X_train, y_train, X_test, y_test)
            best_num_layers = grid_result.best_params_['layers']
            best_batch_norm = grid_result.best_params_['batch_norm']
            best_dropout = grid_result.best_params_['dropout']
            best_epochs = grid_result.best_params_['epochs']
            best_batchsize = grid_result.best_params_['batch_size']
            grid_best_score = grid_result.best_score_
            self.model = self.gridModel(layers=best_num_layers, batch_norm=best_batch_norm, dropout=best_dropout)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
        else:
            self.model, best_hps, param_grid = tuner.tune(X_train, y_train, X_test, y_test)
            # hard coded epochs, batchsize - KerasTuner doesn't search over this parameter space
            best_epochs = 50
            best_batchsize = 10000
            best_num_layers = best_hps.get('num_layers')
            best_batch_norm = best_hps.get('batch_norm')
            best_dropout = best_hps.get('dropout')
            grid_best_score = None
        model = self.train(X_train, X_test, y_train, y_test, epochs=best_epochs, batch_size=best_batchsize, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        if not self.gen:
            file = f'{self.write_dir}/grid_search_{self.channel}.txt'
        else:
            file = f'{self.write_dir}/grid_search_{self.channel}_gen.txt'
        with open(file, 'a+') as f:
            print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            message = f'{time_str},{auc},{self.config_num},{best_num_layers},{best_batch_norm},{best_dropout},{best_epochs},{best_batchsize},{tuning_mode},{grid_best_score},{param_grid}\n'
            print(f"Message: {message}")
            f.write(message)
        # model.save(f'./saved_models/grid_search_model_{config_num}_{self.channel}_{search_mode}/')

    def initialize(self, addons_config={}, read=True, from_pickle=True):
        """
        Initialize NN by loading/ creating the input data for NN via DataLoader
        Params:
        addons(dict) - addon map each value being an addon configuration
        read - will read df inputs instead of creating them
        from_pickle - will read events from pickle instead of .root file
        Returns: df of NN inputs (to be configured) 
        """
        if not addons_config:
            addons = []
        else:
            addons = addons_config.keys()
        if self.channel == 'rho_rho':
            self.DL = DataLoader(config.variables_rho_rho, self.channel)
        elif self.channel == 'rho_a1':
            self.DL = DataLoader(config.variables_rho_a1, self.channel)
        elif self.channel == 'a1_a1':
            self.DL = DataLoader(config.variables_a1_a1, self.channel)
        else:
            raise ValueError('Incorrect channel inputted')
        CC = ConfigChecker(self.channel, self.binary)
        CC.checkInitialize(self.DL, addons_config, read, from_pickle)
        if read:
            print("WARNING: skipping over creating new configs")
            df = self.DL.loadRecoData(self.binary, addons)
        else:
            df = self.DL.createRecoData(self.binary, from_pickle, addons, addons_config)
        return df

    def initializeGen(self, read=True, from_pickle=True):
        if self.channel == 'rho_rho':
            self.DL = DataLoader(config.variables_gen_rho_rho, self.channel)
        elif self.channel == 'rho_a1':
            self.DL = DataLoader(config.variables_gen_rho_a1, self.channel)
        elif self.channel == 'a1_a1':
            self.DL = DataLoader(config.variables_gen_a1_a1, self.channel)
        else:
            raise ValueError('Incorrect channel inputted')
        CC = ConfigChecker(self.channel, self.binary)
        CC.checkInitializeGen(self.DL, read, from_pickle)
        if read:
            df = self.DL.loadGenData(self.binary)
        else:
            df = self.DL.createGenData(self.binary, from_pickle)
        return df

    def configure(self, df, config_num, mode=0):
        """
        Configures NN inputs - selects config_num and creates train/test split
        """
        self.config_num = config_num
        CL = ConfigLoader(df, self.channel)
        X_train, X_test, y_train, y_test = CL.configTrainTestData(self.config_num, self.binary, mode)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=1024, patience=10, save=False, verbose=1):
        self.epochs = epochs
        self.batch_size = batch_size
        if self.model is None:
            self.model = self.kristof_model(X_train.shape[1])
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[self.history, tensorboard_callback], #, early_stop],
                       validation_data=(X_test, y_test),
                       verbose=verbose)
        if save:
            self.model.save(f'./saved_models/{self.save_dir}/NN')
        return self.model

    def evaluate(self, model, X_test, y_test, history, w_a, w_b):
        config_str = self.createConfigStr()
        E = Evaluator(model, self.binary, self.save_dir, config_str)
        auc = E.evaluate(X_test, y_test, history, show=self.show_graph, w_a=w_a, w_b=w_b)
        return auc

    def write(self, auc, history, addons_config):
        if not addons_config:
            addons = []
        else:
            addons = addons_config.keys()
        addons_loaded = "None"
        if addons:
            addons_loaded = '_'+'_'.join(addons)
        file = f'{self.write_dir}/{self.write_filename}.txt'
        with open(file, 'a+') as f:
            print(f'Writing to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            actual_epochs = len(history.history["loss"])
            f.write(f'{time_str},{auc},{self.config_num},{self.layers},{self.epochs},{actual_epochs},{self.batch_size},{self.binary},{self.model_str},{addons_loaded}\n')
        print('Finish writing')
        f.close()

    def writeGen(self, auc, history):
        file = f'{self.write_dir}/{self.write_filename}_gen.txt'
        with open(file, 'a+') as f:
            print(f'Writing to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            actual_epochs = len(history.history["loss"])
            f.write(f'{time_str},{auc},{self.config_num},{self.layers},{self.epochs},{actual_epochs},{self.batch_size},{self.binary},{self.model_str}\n')
        print('Finish writing')
        f.close()

    def evaluateBinary(self, model, X_test, y_test, history):
        config_str = self.createConfigStr()
        E = Evaluator(model, self.binary, self.save_dir, config_str)
        auc = E.evaluate(X_test, y_test, history, show=self.show_graph)
        return auc

    def createConfigStr(self):
        if self.binary:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}_{self.model_str}_binary'
        else:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}_{self.model_str}'
        return config_str

    def seq_model(self, units=(300, 300), batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        self.layers = len(units)
        for unit in units:
            self.model.add(tf.keras.layers.Dense(unit, kernel_initializer='normal'))
            if batch_norm:
                self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            if dropout is not None:
                self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "seq_model"
        return self.model

    def kristof_model(self, dimensions):
        # model by kristof
        model = tf.keras.models.Sequential()
        self.layers = 2
        self.model_str = "kristof_model"
        metrics = ['AUC', 'accuracy']
        model.add(tf.keras.layers.Dense(300, input_dim=dimensions, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(300, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)
        return model

def parser():
    # TODO: epochs, batch_size, write_filename
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', default='rho_rho', choices=['rho_rho', 'rho_a1', 'a1_a1'], help='which channel to load to')
    parser.add_argument('config_num', type=int, help='config num to run on')
    parser.add_argument('-g', '--gen', action='store_true', default=False, help='if load gen data')
    parser.add_argument('-b', '--binary', action='store_false', default=True, help='if learn binary labels')
    parser.add_argument('-t', '--tuning', action='store_true', default=False, help='if tuning is run')
    parser.add_argument('-r', '--read', action='store_false', default=True, help='if read NN input')
    parser.add_argument('-p', '--from_pickle', action='store_false', default=True, help='if read .root file from pickle')
    parser.add_argument('-a', '--addons', nargs='*', default=None, help='load addons')
    parser.add_argument('-s', '--show_graph', action='store_true', default=False, help='if show graphs')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    if not os.path.exists('C:\\Kristof'):  # then we are on Stanley's computer
        # use command line parser - comment out if not needed
        use_parser = True
        if use_parser:
            args = parser()
            channel = args.channel
            config_num = args.config_num 
            gen = args.gen
            binary = args.binary
            tuning = args.tuning
            read = args.read
            from_pickle = args.from_pickle
            addons = args.addons
            show_graph = args.show_graph
            NN = NeuralNetwork(channel=channel, gen=gen, binary=binary, write_filename='NN_output', show_graph=show_graph)
            if not tuning:
                NN.run(config_num, read=read, from_pickle=from_pickle, epochs=50, batch_size=10000)
            else:
                NN.runTuning(config_num, tuning_mode='random_sk')
        else:
            NN = NeuralNetwork(channel='rho_rho', gen=True, binary=True, write_filename='NN_output', show_graph=False)
            # NN.initialize(addons_config={'neutrino': {'load_alpha':False, 'termination':1000}}, read=False, from_pickle=True)
            # NN.initialize(addons_config={}, read=False, from_pickle=True)
            # NN.model = NN.seq_model(units=(300, 300, 300), batch_norm=True, dropout=0.2)
            # NN.run(1, read=True, from_pickle=True, epochs=100, batch_size=8192) # 16384, 131072
            NN.run(3, read=True, from_pickle=True, epochs=50, batch_size=10000)
            # configs = [1,2,3,4,5,6]
            # NN.runMultiple(configs, epochs=1, batch_size=10000)
            # NN.runWithNeutrino(1, load_alpha=False, termination=100, read=False, from_pickle=True, epochs=50, batch_size=1024)
            # NN.runTuning(3, tuning_mode='random_kt')

    else:  # if we are on Kristof's computer
        NN = NeuralNetwork(channel='rho_rho', gen=False, binary=True, write_filename='NN_output', show_graph=False)
        # NN = NeuralNetwork(channel='rho_a1', gen=False, binary=True, write_filename='NN_output', show_graph=False)
        # NN = NeuralNetwork(channel='a1_a1', binary=True, write_filename='NN_output', show_graph=False)
        # NN.run(2, read=True, from_pickle=True, epochs=10, batch_size=10000)
        # NN.run(1, read=False, from_pickle=False, epochs=10, batch_size=10000)
        NN.runTuning(6)
        
        # for i in range(1, 7):
        #     NN.run(i, read=False, from_pickle=False, epochs=50, batch_size=10000)
        #     print()
        #     print()
        #     print('ITERATION', i, 'done')
        #     print()
        #     print()
