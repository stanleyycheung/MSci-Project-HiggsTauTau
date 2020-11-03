import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
from pylorentz import Momentum4
from pylorentz import Position4
from lbn_modified import LBN, LBNLayer
from Task_2 import NN_aco_angle_1 as NN_base


class Potential2016(NN_base):
    def __init__(self):
        super().__init__()
        self.save_dir = "potential_2016"
        self.write_dir = 'potential_2016'
        self.write_filename = 'potential_2016'
        self.layers = 0
    
    def createTrainTestData(self, save=True):
        df = self.df_rho_clean
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        # boost into rest frame of resonances
        rest_frame = pi_1 + pi_2 + pi0_1 + pi0_2
        pi_1_boosted = pi_1.boost_particle(-rest_frame)
        pi_2_boosted = pi_2.boost_particle(-rest_frame)
        pi0_1_boosted = pi0_1.boost_particle(-rest_frame)
        pi0_2_boosted = pi0_2.boost_particle(-rest_frame)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi0_2_boosted
        # find angles of rho vectors to z axis
        unit1 = (rho_1_boosted[1:, :] / np.linalg.norm(rho_1_boosted[1:, :], axis=0)).transpose()
        unit2 = (rho_2_boosted[1:, :] / np.linalg.norm(rho_2_boosted[1:, :], axis=0)).transpose()
        zaxis = np.array([np.array([0., 0., 1.]) for _ in range(len(unit1))])
        angles1 = np.arccos(np.multiply(unit1, zaxis).sum(1))
        angles2 = np.arccos(np.multiply(unit1, zaxis).sum(1))
        axes1 = np.cross(unit1, zaxis)
        axes2 = np.cross(unit2, zaxis)
        # rotations
        pi_1_boosted_rot = []
        pi_2_boosted_rot = []
        pi0_1_boosted_rot = []
        pi0_2_boosted_rot = []
        for i in range(pi_1_boosted[:].shape[1]):
            pi_1_boosted_rot.append(R.from_rotvec(axes1[i]*angles1[i]).apply(pi_1_boosted[1:, i]))
            pi_2_boosted_rot.append(R.from_rotvec(axes2[i]*angles2[i]).apply(pi_2_boosted[1:, i]))
            pi0_1_boosted_rot.append(R.from_rotvec(axes1[i]*angles1[i]).apply(pi0_1_boosted[1:, i]))
            pi0_2_boosted_rot.append(R.from_rotvec(axes2[i]*angles2[i]).apply(pi0_2_boosted[1:, i]))
            if i%100000==0:
                print('finished getting rotated 4-vector', i)
        pi_1_transformed = np.c_[pi_1_boosted[0], np.array(pi_1_boosted_rot)]
        pi_2_transformed = np.c_[pi_2_boosted[0], np.array(pi_2_boosted_rot)]
        pi0_1_transformed = np.c_[pi0_1_boosted[0], np.array(pi0_1_boosted_rot)]
        pi0_2_transformed = np.c_[pi0_2_boosted[0], np.array(pi0_2_boosted_rot)]
        aco_angle_1 = df['aco_angle_1'].to_numpy()
        y_1_1 = df['y_1_1'].to_numpy()
        y_1_2 = df['y_1_2'].to_numpy()
        self.X = np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, aco_angle_1 , y_1_1, y_1_2, rho_1.m, rho_2.m]
        self.y = (self.df_rho['wt_cp_sm']/(self.df_rho['wt_cp_sm'] + self.df_rho['wt_cp_ps'])).to_numpy()
        scaler_x = MinMaxScaler()
        scaler_x.fit(self.X)
        self.X = scaler_x.transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X,self.y,test_size=0.2,random_state=123456,)
        if save:
            print('Saving train/test data to file')
            np.save(f'{self.save_dir}/X.npy', self.X, allow_pickle=True)
            np.save(f'{self.save_dir}/y.npy', self.y, allow_pickle=True)
            print('Saved train/test data')
        return self.X, self.y


    def readTrainTestData(self):
        print("Reading train/test files")
        self.X = np.load(f'{self.save_dir}/X.npy', allow_pickle=True)
        self.y = np.load(f'{self.save_dir}/y.npy', allow_pickle=True)
        print("Loaded train/test files")
        self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X,self.y,test_size=0.2,random_state=123456,)
        return self.X, self.y
    
   
    def createConfigStr(self):
        # TODO: include batch norm
        return f'{self.layers}_{self.epochs}_{self.batch_size}'

    def train(self, epochs=10, batch_size=1000, patience=5):
        # NOT WORKING YET
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.func_model()
        self.config_str = self.createConfigStr()
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.history,early_stop], validation_split=0.3)

    def evaluation(self, write=True, verbose=True):
        # TODO: create custom ROC function
        y_pred = self.model.predict(self.X)
        self.test_loss = self.model.evaluate(self.X, self.y)
        # insert custom ROC function
        if write:
            file = f'{self.write_dir}/{self.write_filename}.txt'
            with open(file, 'a+') as f:
                print('Writing to file')
                f.write(f'{self.loss_function},{self.epochs},{self.batch_size},{self.test_loss[0]},{self.test_loss[1]},{self.mae},{self.mse}\n')
            print('Finish writing')
            f.close()

    def func_model(self, units=[300,300], activations=['relu', 'relu']):
        # TODO: make batch normalisation configurations
        if len(units) != len(activations):
            print('units and activations have different lengths')
            exit(-1)
        self.layers = len(units)
        inputs = tf.keras.Input(shape=(self.X.shape[1],))
        x = inputs
        for i in range(self.layers):
            x = tf.keras.layers.Dense(units[i], activation=activations[i], kernel_initializer='normal')(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC'])
        # print(self.model.summary())
        return self.model


if __name__ == '__main__':
    # set up NN
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Setting up NN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    NN = Potential2016()
    read = True
    if not read:
        NN.readData()
        NN.cleanData()
        NN.createTrainTestData()
    if read:
        NN.readTrainTestData()
    