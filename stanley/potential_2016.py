import uproot 
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import MinMaxScaler
from pylorentz import Momentum4
from lbn_modified import LBN, LBNLayer
from Task_2 import NN_aco_angle_1 as NN_base
from sklearn.metrics import  roc_curve, roc_auc_score


class Potential2016(NN_base):
    def __init__(self):
        super().__init__()
        self.save_dir = "potential_2016"
        self.write_dir = 'potential_2016'
        self.write_filename = 'potential_2016'
        self.file_names = ["pi_1_transformed", "pi_2_transformed", "pi0_1_transformed", "pi0_2_transformed", "rho_1_transformed", "rho_2_transformed", "aco_angle_1", "y_1_1", "y_1_2", "m_1", "m_2", "w_a", "w_b"]
        self.layers = 0
        self.config_num = 0
    
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
        # rotations
        pi_1_boosted_rot = []
        pi_2_boosted_rot = []
        pi0_1_boosted_rot = []
        pi0_2_boosted_rot = []
        rho_1_boosted_rot = []
        rho_2_boosted_rot = []
        for i in range(pi_1_boosted[:].shape[1]):
            rot_mat_1 = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0,0,1])
            rot_mat_2 = self.rotation_matrix_from_vectors(rho_2_boosted[1:, i], [0,0,1])
            pi_1_boosted_rot.append(rot_mat_1.dot(pi_1_boosted[1:, i]))
            pi0_1_boosted_rot.append(rot_mat_1.dot(pi0_1_boosted[1:, i]))
            pi_2_boosted_rot.append(rot_mat_2.dot(pi_2_boosted[1:, i]))
            pi0_2_boosted_rot.append(rot_mat_2.dot(pi0_2_boosted[1:, i]))
            rho_1_boosted_rot.append(rot_mat_1.dot(rho_1_boosted[1:, i]))
            rho_2_boosted_rot.append(rot_mat_2.dot(rho_2_boosted[1:, i]))
            if i%100000==0:
                print('finished getting rotated 4-vector', i)
        self.pi_1_transformed = np.c_[pi_1_boosted[0], np.array(pi_1_boosted_rot)]
        self.pi_2_transformed = np.c_[pi_2_boosted[0], np.array(pi_2_boosted_rot)]
        self.pi0_1_transformed = np.c_[pi0_1_boosted[0], np.array(pi0_1_boosted_rot)]
        self.pi0_2_transformed = np.c_[pi0_2_boosted[0], np.array(pi0_2_boosted_rot)]
        self.rho_1_transformed = np.c_[rho_1_boosted[0], np.array(rho_1_boosted_rot)]
        self.rho_2_transformed = np.c_[rho_2_boosted[0], np.array(rho_2_boosted_rot)]
        self.aco_angle_1 = df['aco_angle_1'].to_numpy()
        self.y_1_1 = df['y_1_1'].to_numpy()
        self.y_1_2 = df['y_1_2'].to_numpy()
        self.w_a = self.df_rho['wt_cp_sm'].to_numpy()
        self.w_b = self.df_rho['wt_cp_ps'].to_numpy()
        self.m_1 = rho_1.m
        self.m_2 = rho_2.m
        if save:
            print('Saving train/test data to file')
            to_save = [self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b]
            for i in range(len(to_save)):
                np.save(f'{self.save_dir}/{self.file_names[i]}', to_save[i], allow_pickle=True)
            print('Saved train/test data')

    def readTrainTestData(self):
        print("Reading train/test files")
        to_load = []
        for i in range(len(self.file_names)):
            to_load.append(np.load(f'{self.save_dir}/{self.file_names[i]}.npy', allow_pickle=True))
        self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b = to_load
        print("Loaded train/test files")

    def configTrainTestData(self, config_num):
        self.config_num = config_num
        config_map = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1 , self.y_1_1, self.y_1_2, self.m_1, self.m_2],
        }
        try:
            self.X = config_map[self.config_num]
        except KeyError as e:
            print('Wrong config input number')
            exit(-1)
        self.y = (self.w_a/(self.w_a+self.w_b))
        # scaler_x = MinMaxScaler()
        # scaler_x.fit(self.X)
        # self.X = scaler_x.transform(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X, self.y, test_size=0.2, random_state=123456,)
        return self.X_train, self.X_test, self.y_train, self.y_test
   
    def createConfigStr(self):
        # TODO: include batch norm
        return f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}'

    def train(self, epochs=10, batch_size=1000, patience=5):
        # NOT WORKING YET
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.func_model()
        self.config_str = self.createConfigStr()
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[self.history,early_stop],
                        validation_data=(self.X_test, self.y_test))
        # self.model.save(f'{self.save_dir}/NN_1')

    def evaluation(self, write=True):
        y_pred = self.model.predict(self.X)
        # creating custom ROC
        set_a = np.ones(len(y_pred))
        set_b = np.zeros(len(y_pred))
        y_pred_roc = np.r_[y_pred, y_pred]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[self.w_a, self.w_b]
        custom_auc = roc_auc_score(y_label_roc, y_pred_roc, sample_weight=w_roc)
        fpr, tpr, _ = roc_curve(y_label_roc, y_pred_roc, sample_weight=w_roc)
        self.plot_roc_curve(fpr, tpr, custom_auc)
        if write:
            file = f'{self.write_dir}/{self.write_filename}.txt'
            with open(file, 'a+') as f:
                print('Writing to file')
                f.write(f'{custom_auc},{self.config_num},{self.layers},{self.epochs},{self.batch_size}\n')
            print('Finish writing')
            f.close()

    def func_model(self, units=[300,300], activations=['relu', 'relu']):
        # TODO: make batch normalisation configurations and dropout
        # paper uses batch normalisation before activation
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
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        # print(self.model.summary())
        return self.model

    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def plot_roc_curve(self, fpr, tpr, auc):
        #  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.grid()
        ax.text(0.6, 0.3, 'Custom AUC Score: {:.3f}'.format(auc),
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.savefig(f'{self.save_dir}/fig/ROC_curve_{self.config_str}.PNG')

    def plotLoss(self):
        plt.clf()
        # Extract number of run epochs from the training history
        epochs = range(1, len(self.history.history["loss"])+1)
        # Extract loss on training and validation ddataset and plot them together
        plt.figure(figsize=(10,8))
        plt.title(f'loss_fn:{self.loss_function}')
        plt.plot(epochs, self.history.history["loss"], "o-", label="Training")
        plt.plot(epochs, self.history.history["val_loss"], "o-", label="Test")
        plt.xlabel("Epochs"), plt.ylabel("Loss")
        # plt.yscale("log")
        plt.legend()
        plt.savefig(f'./{self.save_dir}/fig/loss_{self.config_str}')
        plt.show()

if __name__ == '__main__':
    # set up NN
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Setting up NN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    NN = Potential2016()
    read = True
    if not read:
        NN.readData()
        NN.cleanData()
        NN.createTrainTestData()
    else:
        NN.readTrainTestData()
    # NN.configTrainTestData(6)
    # NN.train(epochs=10, batch_size=1000)
    # NN.evaluation(write=True)
    # NN.plotLoss()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished NN setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    for i in range(1, 7):
        NN.configTrainTestData(i)
        NN.train(epochs=50, batch_size=1000)
        NN.evaluation(write=True)
        NN.plotLoss()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')