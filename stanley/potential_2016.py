# set same seed
from datetime import datetime
from sklearn.metrics import roc_curve, roc_auc_score
from NN_base import NN_base
from pylorentz import Momentum4
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import os
seed_value = 0
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
# for later versions:
tf.compat.v1.set_random_seed(seed_value)


class Potential2016(NN_base):
    def __init__(self, binary, alt_label=False, write_filename='potential_2016'):
        super().__init__()
        self.save_dir = "potential_2016"
        self.load_dir = "potential_2016"
        self.write_dir = 'potential_2016'
        self.file_names = ["pi_1_transformed", "pi_2_transformed", "pi0_1_transformed", "pi0_2_transformed",
                           "rho_1_transformed", "rho_2_transformed", "aco_angle_1", "y_1_1", "y_1_2",
                           "m_1", "m_2", "w_a", "w_b", "E_miss", "E_miss_x", "E_miss_y",
                           "aco_angle_5", "aco_angle_6", "aco_angle_7",
                           "y"]
        self.layers = 0
        self.config_num = 0
        self.binary = binary
        self.write_filename = write_filename
        self.model_str = None
        self.alt_label = alt_label
        if self.binary:
            self.write_filename += '_binary'

    def createTrainTestData(self, save=True):
        if self.binary:
            y_sm = pd.DataFrame(np.ones(self.df_rho_sm.shape[0]))
            y_ps = pd.DataFrame(np.zeros(self.df_rho_ps.shape[0]))
            self.y = pd.concat([y_sm, y_ps]).to_numpy()
            df = pd.concat([self.df_rho_sm, self.df_rho_ps]).drop(["wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
                                                                   "tau_decay_mode_1", "tau_decay_mode_2", "mva_dm_1", "mva_dm_2", ], axis=1).reset_index(drop=True)
        else:
            df = self.df_rho_clean
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"],
                         df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"],
                         df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"],
                          df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"],
                          df["pi0_py_2"], df["pi0_pz_2"])
        N = len(df['metx'])
        met_x = Momentum4(df['metx'], np.zeros(N), np.zeros(N), np.zeros(N))
        met_y = Momentum4(df['mety'], np.zeros(N), np.zeros(N), np.zeros(N))
        met = Momentum4(df['met'], np.zeros(N), np.zeros(N), np.zeros(N))
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        # boost into rest frame of resonances
        rest_frame = pi_1 + pi_2 + pi0_1 + pi0_2
        boost = Momentum4(
            rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi0_2_boosted = pi0_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi0_2_boosted
        # boost MET - E_miss is already boosted into the hadronic rest frame
        self.E_miss = met_x.boost_particle(boost)[0]
        self.E_miss_x = met_y.boost_particle(boost)[0]
        self.E_miss_y = met.boost_particle(boost)[0]
        # rotations
        pi_1_boosted_rot = []
        pi_2_boosted_rot = []
        pi0_1_boosted_rot = []
        pi0_2_boosted_rot = []
        rho_1_boosted_rot = []
        rho_2_boosted_rot = []
        for i in range(pi_1_boosted[:].shape[1]):
            rot_mat = self.rotation_matrix_from_vectors(
                rho_1_boosted[1:, i], [0, 0, 1])
            pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
            pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
            pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
            pi0_2_boosted_rot.append(rot_mat.dot(pi0_2_boosted[1:, i]))
            rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
            rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
            if i % 100000 == 0:
                print('finished getting rotated 4-vector', i)
        self.pi_1_transformed = np.c_[
            pi_1_boosted[0], np.array(pi_1_boosted_rot)]
        self.pi_2_transformed = np.c_[
            pi_2_boosted[0], np.array(pi_2_boosted_rot)]
        self.pi0_1_transformed = np.c_[
            pi0_1_boosted[0], np.array(pi0_1_boosted_rot)]
        self.pi0_2_transformed = np.c_[
            pi0_2_boosted[0], np.array(pi0_2_boosted_rot)]
        self.rho_1_transformed = np.c_[
            rho_1_boosted[0], np.array(rho_1_boosted_rot)]
        self.rho_2_transformed = np.c_[
            rho_2_boosted[0], np.array(rho_2_boosted_rot)]
        self.aco_angle_1 = df['aco_angle_1'].to_numpy()
        self.aco_angle_5 = df['aco_angle_5'].to_numpy()
        self.aco_angle_6 = df['aco_angle_6'].to_numpy()
        self.aco_angle_7 = df['aco_angle_7'].to_numpy()
        self.y_1_1 = df['y_1_1'].to_numpy()
        self.y_1_2 = df['y_1_2'].to_numpy()
        self.w_a = self.df_rho['wt_cp_sm'].to_numpy()
        self.w_b = self.df_rho['wt_cp_ps'].to_numpy()
        self.m_1 = rho_1.m
        self.m_2 = rho_2.m
        to_save = [self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed,
                   self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2,
                   self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y,
                   self.aco_angle_5, self.aco_angle_6, self.aco_angle_7]
        extra_to_save = self.createExtraData(df, boost)
        if extra_to_save is not None:
            to_save += extra_to_save
            print('Extra data loaded')

        if save:
            print('Saving train/test data to file')
            if self.binary:
                to_save += [self.y]
            for i in range(len(to_save)):
                if self.binary:
                    save_name = f'{self.save_dir}/{self.file_names[i]}_b'
                else:
                    save_name = f'{self.save_dir}/{self.file_names[i]}'
                np.save(save_name, to_save[i], allow_pickle=True)
                print(f"Saving {save_name}")
            print('Saved train/test data')

    def readTrainTestData(self):
        print("Reading train/test files")
        to_load = []
        if not self.binary:
            self.file_names.pop()
        for i in range(len(self.file_names)):
            if self.binary:
                load_name = f'{self.load_dir}/{self.file_names[i]}_b.npy'
            else:
                load_name = f'{self.load_dir}/{self.file_names[i]}.npy'
            to_load.append(np.load(load_name, allow_pickle=True))
        if self.binary:
            self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7, self.y = to_load
        else:
            self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7 = to_load
        print("Loaded train/test files")

    def chooseConfigMap(self, mode=1):
        config_map_orig = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            7: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
        }
        config_map_norho = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            # adding extra aco angles
            7: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7],
            8: np.c_[self.aco_angle_5, self.aco_angle_6, self.aco_angle_7],
            9: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7],
            10: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.E_miss],
            11: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7, self.E_miss],
        }
        config_map_onlyrho = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.rho_1_transformed, self.rho_2_transformed],
            4: np.c_[self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
        }
        mode_map = [config_map_orig, config_map_norho, config_map_onlyrho, ]
        additional_configs = self.addConfigs()
        if additional_configs is not None:
            mode_map += additional_configs
        print(
            f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in mode {mode}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return mode_map[mode]

    def configTrainTestData(self, config_num, mode=1):
        self.config_num = config_num
        try:
            config_map = self.chooseConfigMap(mode=mode)
        except KeyError as e:
            print('Wrong config input number')
            exit(-1)
        self.X = config_map[self.config_num]
        if self.binary:
            # self.y.astype(int) # probably doesn't matter
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=123456, stratify=self.y,)
        else:
            if self.alt_label:
                self.y = (self.w_a > self.w_b).astype(int)
            else:
                self.y = (self.w_a/(self.w_a+self.w_b))
            # self.y = np.load('./potential_2016/y_kristof.npy', allow_pickle=True) #kristof's method
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=123456,)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def createConfigStr(self):
        if self.binary:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}_{self.model_str}_binary'
        else:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}_{self.model_str}'
        return config_str

    def trainRepeated(self, repeated_num, config, arch_str, time_str, external_model, epochs, batch_size, patience=10):
        auc_score_arr = []
        for i in range(repeated_num):
            print(
                f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training {i+1} out of {repeated_num} times~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            self.seq_model(*config)
            self.train(external_model=external_model, epochs=epochs,
                       batch_size=batch_size, patience=patience)
            auc_score = self.evaluation(write=False)
            auc_score_arr.append(auc_score)
        auc_avg = np.average(auc_score_arr)
        if repeated_num == 1:
            auc_error = 0
        else:
            auc_error = np.std(auc_score_arr, ddof=1)
        self.writeMessage(
            f'{time_str}-{arch_str}-{auc_avg}-{auc_error}-{self.config_num}-{self.layers}-{self.epochs}-{self.batch_size}-{self.binary}')

    def train(self, external_model=False, epochs=50, batch_size=1024, patience=10):
        self.epochs = epochs
        self.batch_size = batch_size
        # if model_num == 1:
        #     self.model = self.seq_model()
        # elif model_num == 2:
        if not external_model:
            self.model = self.kristof_model(self.X.shape[1])
        # elif model_num == 3:
        #     units = [300]*6
        #     activations = ['relu']*6
        #     self.model = self.func_model(units, activations)
        self.config_str = self.createConfigStr()
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[self.history, early_stop],
                       validation_data=(self.X_test, self.y_test))
        # self.model.save(f'{self.save_dir}/NN_1')

    def evaluation(self, write=True):
        # use test dataset for evaluation
        if self.binary:
            y_proba = self.model.predict_proba(
                self.X_test)  # outputs two probabilties
            # print(y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            self.plot_roc_curve(fpr, tpr, auc)
        else:
            y_pred_test = self.model.predict(self.X_test)
            _, w_a_test, _, w_b_test = train_test_split(
                self.w_a, self.w_b, test_size=0.2, random_state=123456)
            auc, y_label_roc, y_pred_roc = self.custom_auc_score(
                y_pred_test, w_a_test, w_b_test)
            fpr, tpr, _ = roc_curve(
                y_label_roc, y_pred_roc, sample_weight=np.r_[w_a_test, w_b_test])
            self.plot_roc_curve(fpr, tpr, auc)

        if write:
            file = f'{self.write_dir}/{self.write_filename}.txt'
            with open(file, 'a+') as f:
                print(f'Writing to {file}')
                f.write(
                    f'{auc},{self.config_num},{self.layers},{self.epochs},{self.batch_size},{self.binary},{self.model_str}\n')
            print('Finish writing')
            f.close()
        return auc

    def custom_auc_score(self, pred, w_a, w_b):
        set_a = np.ones(len(pred))
        set_b = np.zeros(len(pred))
        y_pred_roc = np.r_[pred, pred]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[w_a, w_b]
        custom_auc = roc_auc_score(
            y_label_roc, y_pred_roc, sample_weight=w_roc)
        return custom_auc, y_label_roc, y_pred_roc

    def func_model(self, units=[300, 300, 300], activations=['relu', 'relu', 'relu'], batch_norm=False, dropout=False):
        # TODO: make batch normalisation configurations and dropout, paper uses batch normalisation before activation
        if len(units) != len(activations):
            print('units and activations have different lengths')
            exit(-1)
        self.layers = len(units)
        inputs = tf.keras.Input(shape=(self.X.shape[1],))
        x = inputs
        for i in range(self.layers):
            x = tf.keras.layers.Dense(
                units[i], activation=None, kernel_initializer='normal')(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(activations[i])(x)
            if dropout:
                x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model_str = "func_model"
        # print(self.model.summary())
        return self.model

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
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model_str = "seq_model"
        return self.model

    def kristof_model(self, dimensions=-1):
        # model by kristof
        if dimensions == -1:
            dimensions = self.X.shape[1]
        # create model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(
            38, input_dim=dimensions, kernel_initializer='normal', activation='relu'))
        self.model.add(tf.keras.layers.Dense(
            100, kernel_initializer='normal', activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model_str = "kristof_model"
        return self.model

    def writeMessage(self, message, file=None):
        if file is None:
            file = f'{self.write_dir}/{self.write_filename}.txt'
        with open(file, 'a+') as f:
            print(f'Writing {message} to file')
            f.write(f'{message}\n')
        print('Finish writing')
        f.close()

    def writeDateTime(self, file=None):
        if file is None:
            file = f'{self.write_dir}/{self.write_filename}.txt'
        message = '#' + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(file, 'a+') as f:
            print(f'Writing {message} to file')
            f.write(f'{message}\n')
        print('Finish writing')
        f.close()

    def aucTest(self):
        # TODO: change w_roc
        # theroetical limit (non-binary labels)
        set_a = np.ones(len(self.y))
        set_b = np.zeros(len(self.y))
        y_pred_roc = np.r_[self.y, self.y]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[self.w_a, self.w_b]
        custom_auc = roc_auc_score(
            y_label_roc, y_pred_roc, sample_weight=w_roc)
        print(f'Theoretical limit:{custom_auc}')
        # calc auc for both test and train dataset
        # y_pred_test = self.model.predict(self.X_test)
        # y_pred_train = self.model.predict(self.X_train)
        # # creating custom ROC
        # y_pred_roc_test = np.r_[y_pred_test, y_pred_test]
        # y_pred_roc_train = np.r_[y_pred_train, y_pred_train]
        # custom_auc_test = roc_auc_score(np.r_[np.ones(len(y_pred_test)), np.zeros(len(y_pred_test))], y_pred_roc_test, sample_weight=w_roc)
        # custom_auc_train = roc_auc_score(np.r_[np.ones(len(y_pred_train)), np.zeros(len(y_pred_train))], y_pred_roc_train, sample_weight=w_roc)
        # print(f'Test AUC score: {custom_auc_test}')
        # print(f'Train AUC score: {custom_auc_train}')

    def addConfigs(self):
        return []

    def createExtraData(self, df, boost):
        return []


def initNN(NN, read=True):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Setting up NN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    if not read:
        NN.readData(from_pickle=False)
        NN.cleanData()
        NN.createTrainTestData()
    else:
        NN.readTrainTestData()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished NN setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def runNN(NN, config_num, epochs, batch_size, mode=1):
    print(
        f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {config_num}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    NN.seq_model()
    NN.configTrainTestData(config_num, mode)
    NN.train(external_model=True, epochs=epochs, batch_size=batch_size)
    # NN.aucTest()
    NN.evaluation(write=True)
    # NN.plotLoss()
    plt.close()


def runConfigsNN(NN, start, end, epochs=50, batch_size=10000, mode=1):
    # TODO: accept arb NN config
    for i in range(start, end+1):
        print(
            f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # NN.writeMessage('')
        NN.configTrainTestData(i, mode)
        NN.seq_model()
        NN.train(external_model=True, epochs=epochs, batch_size=batch_size)
        NN.evaluation(write=True)
        NN.plotLoss()
        plt.close()


def runArchitecturesNN(NN, NN_config, repeated_num, epochs=20, batch_size=2048, mode=1):
    # NN.writeDateTime()
    time_str = datetime.now().strftime('%Y/%m/%d|%H:%M:%S')
    for config in NN_config:
        print(
            f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training {config} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        arch_str = '-'.join([str(x) for x in config])
        config_to_train = [3, 6]
        for i in config_to_train:
            print(
                f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            NN.configTrainTestData(i, mode)
            NN.trainRepeated(repeated_num, config, arch_str, time_str,
                             external_model=True, epochs=epochs, batch_size=batch_size, patience=7)
            plt.close()


def getNN_config(layers, dropout=0.2):
    layer_config = [300]*layers
    return [[layer_config, False, None],
            [layer_config, True, None],
            [layer_config, False, dropout],
            [layer_config, True, dropout], ]


def runScriptArchitectures():
    NN = Potential2016(binary=True, write_filename='potential_2016_archs')
    initNN(NN, read=True)
    NN_config = getNN_config(6)
    runArchitecturesNN(NN, NN_config, 1, epochs=50, batch_size=100000)


if __name__ == '__main__':
    NN = Potential2016(binary=True, write_filename='potential_2016')
    # NN = Potential2016(binary=False, alt_label=True, write_filename='potential_2016')
    # NN = Potential2016(binary=True, write_filename='potential_2016_archs')
    # set up NN
    initNN(NN, read=False)
    # runNN(NN, 3, 50, 100000)
    # runNN(NN, 3, 50, 100000)
    # NN_config = getNN_config(6)
    # NN_config = [[[300]*6, True, 0.2]]
    # runArchitecturesNN(NN, NN_config, 1, epochs=200, batch_size=100000)
    # runConfigsNN(NN, 1, 11, epochs=100, batch_size=10000)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
