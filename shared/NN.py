from numpy.core.numeric import True_
from evaluator import Evaluator
from data_loader import DataLoader
from config_loader import ConfigLoader
from config_checker import ConfigChecker
import os
import tensorflow as tf
import random
import numpy as np
import datetime
import kerastuner as kt
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
seed_value = 1
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
# os.environ['PYTHONHASHSEED'] = str(seed_value)
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
# random.seed(seed_value)
# # 3. Set the `numpy` pseudo-random generator at a fixed value
# np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
# tf.compat.v1.set_random_seed(seed_value)


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
            'metcov00', 'metcov01', 'metcov10', 'metcov11',
            "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
            "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2" #subleading neutrino, gen level
        ]
        self.variables_rho_a1 = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1",
            "mva_dm_1", "mva_dm_2",
            "tau_decay_mode_1", "tau_decay_mode_2",
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
            "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
            "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
            "ip_x_1", "ip_y_1", "ip_z_1",
            "sv_x_2", "sv_y_2", "sv_z_2",
            "y_1_1", "y_1_2",
        ]
        self.variables_a1_a1 = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1",
            "mva_dm_1", "mva_dm_2",
            "tau_decay_mode_1", "tau_decay_mode_2",
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
            "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
            "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
            "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
            "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
            "ip_x_1", "ip_y_1", "ip_z_1",
            "sv_x_2", "sv_y_2", "sv_z_2",
            "y_1_1", "y_1_2",
        ]
        self.save_dir = 'NN_output'
        self.write_dir = 'NN_output'
        self.model = None

    def run(self, config_num, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10, addons_config={}):
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        self.write(auc, self.history, addons_config)

    def runWithNeutrino(self, config_num, load_alpha=False, termination=1000, read=True, from_pickle=True, epochs=50, batch_size=1024, patience=10):
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
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
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
            self.write(auc, self.history, addons_config)

    def runHPTuning(self, config_num, read=True, from_pickle=True, epochs=50, tuner_epochs=50, batch_size=10000, tuner_batch_size=10000, patience=10, tuner_mode=0, addons_config={}):
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tuning on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        best_hps, tuner = self.tuneHP(self.hyperModel, X_train, X_test, y_train, y_test, tuner_epochs=tuner_epochs, tuner_batch_size=tuner_batch_size, tuner_mode=tuner_mode)
        print(tuner.results_summary())
        model = tuner.hypermodel.build(best_hps)
        self.model = model
        # model.fit(X_train, y_train, epochs=epochs, validation_data = (X_test, y_test), verbose=0)
        # verbose is set 0 to prevent logs from spam
        model = self.train(X_train, X_test, y_train, y_test, epochs=epochs, batch_size=batch_size, patience=patience, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        file = f'{self.write_dir}/best_hp_{self.channel}.txt'
        with open(file, 'a+') as f:
            # print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            best_num_layers = best_hps.get('num_layers')
            # best_batch_norm = best_hps.get('batch_norm')
            # best_dropout = best_hps.get('dropout')
            actual_epochs = len(self.history.history["loss"])
            # message = f'{time_str},{auc},{self.config_num},{best_num_layers},{best_batch_norm},{best_dropout},{tuner_mode}\n'
            message = f'{time_str},{auc},{self.config_num},{best_num_layers},{tuner_mode}\n'
            print(f"Message: {message}")
        #     f.write(message)
        # model.save('./hp_model_1/')

    def runGridSearch(self, config_num, read=True, from_pickle=True, addons_config={}, search_mode=0):
        """
        Runs grid search on NN with given config_num
        search_mode = 0: GridSearch
        search_mode = 1: RandomSearch
        """
        df = self.initialize(addons_config, read=read, from_pickle=from_pickle)
        X_train, X_test, y_train, y_test = self.configure(df, config_num)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Grid searching on config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        model = KerasClassifier(self.gridModel, verbose=0)
        if search_mode == 0:
            layers = [2,3,4,5,6]
            batch_norms = [True, False]
            dropouts = [None, 0.2]
            epochs = [100, 200, 500]
            batch_sizes = [8192, 16384, 65536, 131072]
            param_grid = dict(
                layers=layers,
                batch_norm=batch_norms,
                dropout=dropouts,
                epochs=epochs,
                batch_size=batch_sizes
            )
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2, verbose=2, scoring='roc_auc')
        elif search_mode == 1:
            layers = np.arange(1,11)
            batch_norms = [True, False]
            dropouts = [None, 0.1, 0.2, 0.3, 0.4, 0.5]
            epochs = [50, 100, 200, 500]
            batch_sizes = [2**i for i in range(8, 19)]
            # layers = [2,3]
            # batch_norms = [True, False]
            # dropouts = [None, 0.2]
            # epochs = [200, 500]
            # batch_sizes = [8192, 16384, 65536, 131072]
            param_grid = dict(
                layers=layers,
                batch_norm=batch_norms,
                dropout=dropouts,
                epochs=epochs,
                batch_size=batch_sizes
            )
            # can increase the distributions of params
            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, cv=2, verbose=2, scoring='roc_auc', random_state=seed_value, n_iter=20)
        else:
            raise ValueError('Search mode not defined correctly')
        grid_result = grid.fit(X_train, y_train)
        print(grid_result)
        best_num_layers = grid_result.best_params_['layers']
        best_batch_norm = grid_result.best_params_['batch_norm']
        best_dropout = grid_result.best_params_['dropout']
        best_epochs = grid_result.best_params_['epochs']
        best_batchsize = grid_result.best_params_['batch_size']
        self.model = self.gridModel(layers=best_num_layers, batch_norm=best_batch_norm, dropout=best_dropout)
        model = self.train(X_train, X_test, y_train, y_test, epochs=best_epochs, batch_size=best_batchsize, verbose=0)
        if self.binary:
            auc = self.evaluateBinary(model, X_test, y_test, self.history)
        else:
            w_a = df.w_a
            w_b = df.w_b
            auc = self.evaluate(model, X_test, y_test, self.history, w_a, w_b)
        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
        file = f'{self.write_dir}/grid_search_{self.channel}.txt'
        with open(file, 'a+') as f:
            print(f'Writing HPs to {file}')
            time_str = datetime.datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
            message = f'{time_str},{auc},{self.config_num},{best_num_layers},{best_batch_norm},{best_dropout},{best_epochs},{best_batchsize},{search_mode},{grid_result.best_score_},{param_grid}\n'
            print(f"Message: {message}")
            f.write(message)
        model.save(f'./saved_models/grid_search_model_{config_num}_{self.channel}_{search_mode}/')

    def tuneHP(self, hyperModel, X_train, X_test, y_train, y_test, tuner_epochs=50, tuner_batch_size=10000, tuner_mode=0):
        if tuner_mode == 0:
            tuner = kt.Hyperband(hyperModel,
                                 objective=kt.Objective("auc", direction="max"),  # ['loss', 'auc', 'accuracy', 'val_loss', 'val_auc', 'val_accuracy']
                                 max_epochs=200,
                                 hyperband_iterations=3,
                                 factor=3,
                                 seed=seed_value,
                                 directory='tuning',
                                 project_name='model_hyperband_1',
                                 overwrite=True)
        elif tuner_mode == 1:
            tuner = kt.BayesianOptimization(hyperModel,
                                            objective='val_loss',
                                            max_trials=100,
                                            seed=seed_value,
                                            directory='tuning',
                                            project_name='model_bayesian_1',
                                            overwrite=True)
        elif tuner_mode == 2:
            tuner = kt.RandomSearch(hyperModel,
                                    objective='val_loss',
                                    max_trials=1000,
                                    seed=seed_value,
                                    directory='tuning',
                                    project_name='model_random_1',
                                    overwrite=True)
        else:
            raise ValueError('Invalid tuner mode')
        tuner.search(X_train, y_train, epochs=tuner_epochs, batch_size=tuner_batch_size, validation_data=(X_test, y_test), verbose=0)
        # tuner.search(X_train, y_train, epochs=tuner_epochs, validation_data=(X_test, y_test), verbose=1)
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(tuner.search_space_summary())
        return best_hps, tuner

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
            self.DL = DataLoader(self.variables_rho_rho, self.channel)
        elif self.channel == 'rho_a1':
            self.DL = DataLoader(self.variables_rho_a1, self.channel)
        elif self.channel == 'a1_a1':
            self.DL = DataLoader(self.variables_a1_a1, self.channel)
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

    def hyperModel(self, hp):
        self.model = tf.keras.models.Sequential()
        num_layers = hp.Int('num_layers', 2, 3)
        self.layers = num_layers
        for i in range(num_layers):
            self.model.add(tf.keras.layers.Dense(units=300, kernel_initializer='normal'))
            # if hp.Boolean('batch_norm', default=False):
                # self.model.add(tf.keras.layers.BatchNormalization())
            # self.model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.2, default=0.0, step=0.2)))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "hyper_model"
        return self.model

    def gridModel(self, layers=2, batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        self.layers = layers
        for i in range(layers):
            self.model.add(tf.keras.layers.Dense(300, kernel_initializer='normal'))
            if batch_norm:
                self.model.add(tf.keras.layers.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            if dropout is not None:
                self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        metrics = ['AUC', 'accuracy']
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
        self.model_str = "grid_model"
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

def runGridSearchOverConfigs(search_mode, start=1, end=6):
    print(f'Grid searching with search_mode={search_mode}')
    for i in range(start, end+1):
        print(f'Grid searching over config {i}')
        NN.runGridSearch(6, read=True, from_pickle=True, search_mode=search_mode)


if __name__ == '__main__':
    if not os.path.exists('C:\\Kristof'):  # then we are on Stanley's computer
        NN = NeuralNetwork(channel='rho_rho', binary=True, write_filename='NN_output', show_graph=False)
        # NN.initialize(addons_config={'neutrino': {'load_alpha':False, 'termination':1000}}, read=False, from_pickle=True)
        # NN.initialize(addons_config={}, read=False, from_pickle=True)
        # NN.model = NN.seq_model(units=(300, 300, 300), batch_norm=True, dropout=0.2)
        NN.run(1, read=True, from_pickle=True, epochs=100, batch_size=8192) # 16384, 131072
        # configs = [1,2,3,4,5,6]
        # NN.runMultiple(configs, epochs=1, batch_size=10000)
        # NN.runWithNeutrino(1, load_alpha=False, termination=100, read=False, from_pickle=True, epochs=50, batch_size=1024)
        # NN.runHPTuning(3, read=True, from_pickle=True, epochs=200, tuner_epochs=200, batch_size=8192, tuner_batch_size=8192, tuner_mode=1)
        # NN.runGridSearch(6, read=True, from_pickle=True, search_mode=1)
        # runGridSearchOverConfigs(1)

    else:  # if we are on Kristof's computer
        NN = NeuralNetwork(channel='rho_rho', binary=True, write_filename='NN_output', show_graph=False)
        # NN = NeuralNetwork(channel='rho_a1', binary=True, write_filename='NN_output', show_graph=False)
        # NN = NeuralNetwork(channel='a1_a1', binary=True, write_filename='NN_output', show_graph=False)
        # NN.run(3, read=True, from_pickle=True, epochs=25, batch_size=10000)
        NN.run(1, read=False, from_pickle=False, epochs=10, batch_size=10000)
        
        # for _ in range(7):
        #     NN.run(1, read=False, from_pickle=False, epochs=10, batch_size=10000)
        #     print()
        #     print()
        #     print('ITERATION', _, 'done')
        #     print()
        #     print()
