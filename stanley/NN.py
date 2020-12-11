from evaluator import Evaluator
from data_loader import DataLoader
from config_loader import ConfigLoader
import tensorflow as tf
import os
import random
import numpy as np
import datetime
import kerastuner as kt
seed_value = 1
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

    def __init__(self,  channel, binary, write_filename, show_graph=False):
        self.show_graph = show_graph
        self.channel = channel
        self.binary = binary
        self.write_filename = write_filename
        # variables are bare bones only for rho_rho
        # TODO: make variables for all channels neatly in data loader
        self.variables_rho_rho = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1",
            "mva_dm_1", "mva_dm_2",
            "tau_decay_mode_1", "tau_decay_mode_2",
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
            "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
            "y_1_1", "y_1_2",
            'met', 'metx', 'mety',
        ]
        self.save_dir = 'potential_2016'
        self.write_dir = 'potential_2016'

    def run(self, config_num, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10, external_model=False, addons=[]):
        df = self.initalize(addons, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience, external_model=external_model)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        self.write(auc)

    def runMultiple(self, configs, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10, external_model=False, addons=[]):
        df = self.initalize(addons, read=read, from_pickle=from_pickle)
        for config_num in configs:
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            X_train, X_test, y_train, y_test = self.configure(df, config_num)
            model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience, external_model=external_model)
            if self.binary:
                auc = self.evaluateBinary(model, X_test, y_test, self.history)
            else:
                w_a = df.w_a
                w_b = df.w_b
                auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
            self.write(auc)

    def runHPTuning(self, config_num, read=True, from_pickle=True, epochs=50, tuner_epochs=50, addons=[]):
        df = self.initalize(addons, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tuning on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        best_hps, tuner = self.tuneHP(self.hyperModel, X_train, X_test, y_train, y_test, tuner_epochs=tuner_epochs)
        print(best_hps)
        model = tuner.hypermodel.build(best_hps[0])
        model.fit(X_train, y_train, epochs=epochs, validation_data = (X_test, y_test))


    def tuneHP(self, hyperModel, X_train, X_test, y_train, y_test, tuner_epochs=10):
        tuner = kt.Hyperband(hyperModel,
                             objective='val_accuracy',
                             max_epochs=50,
                             factor=3,
                             seed=seed_value,
                             directory='tuning',
                             project_name='test')
        tuner.search(X_train, y_train, epochs=tuner_epochs, validation_data=(X_test, y_test))
        best_hps = tuner.get_best_hyperparameters(num_trials=1)
        print(tuner.search_space_summary())
        return best_hps, tuner

    def initalize(self, addons=[], read=True, from_pickle=True):
        self.DL = DataLoader(self.variables_rho_rho, self.channel)
        if read:
            df = self.DL.loadRecoData(self.binary)
        else:
            df = self.DL.createRecoData(self.binary, from_pickle, addons)
        return df

    def configure(self, df, config_num):
        self.config_num = config_num
        CL = ConfigLoader(df)
        X_train, X_test, y_train, y_test = CL.configTrainTestData(self.config_num, self.binary)
        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test, epochs=50, batch_size=1024, patience=10, external_model=False, save=False):
        self.epochs = epochs
        self.batch_size = batch_size
        if not external_model:
            self.model = self.kristof_model(X_train.shape[1])
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        self.model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       callbacks=[self.history, early_stop, tensorboard_callback],
                       validation_data=(X_test, y_test))
        if save:
            self.model.save(f'{self.save_dir}/NN')
        return self.model

    def evaluate(self, model, X_test, y_test, history, w_a, w_b):
        config_str = self.createConfigStr()
        E = Evaluator(model, self.binary, self.save_dir, config_str)
        auc = E.evaluate(X_test, y_test, history, show=self.show_graph, w_a=w_a, w_b=w_b)
        return auc

    def write(self, auc):
        file = f'{self.write_dir}/{self.write_filename}.txt'
        with open(file, 'a+') as f:
            print(f'Writing to {file}')
            f.write(f'{auc},{self.config_num},{self.layers},{self.epochs},{self.batch_size},{self.binary},{self.model_str}\n')
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

    def seq_model(self, units=[300, 300], batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        self.layers = len(units)
        for unit in units:
            self.model.add(tf.keras.layers.Dense(
                unit, kernel_initializer='normal'))
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

    def hyperModel(self, hp):
        self.model = tf.keras.models.Sequential()
        hp_units = hp.Int('units', min_value=128, max_value=256, step=64)
        self.layers = hp_units
        self.model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dense(units=hp_units, activation='relu', kernel_initializer='normal'))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "hyper_model"
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
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        return model


if __name__ == '__main__':
    NN = NeuralNetwork(channel='rho_rho', binary=True, write_filename='potential_2016', show_graph=False)
    NN.run(3, read=True, from_pickle=True, epochs=50, batch_size=10000)
    # configs = [1,2,3,4,5,6]
    # NN.runMultiple(configs, epochs=1, batch_size=10000)
    # NN.runHPTuning(3, read=True, from_pickle=True, epochs=50, tuner_epochs=50)
