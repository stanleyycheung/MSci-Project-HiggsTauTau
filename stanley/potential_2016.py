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
    def __init__(self, binary):
        super().__init__()
        self.save_dir = "potential_2016"
        self.load_dir = "potential_2016"
        self.write_dir = 'potential_2016'
        self.write_filename = 'potential_2016_binary'
        self.file_names = ["pi_1_transformed", "pi_2_transformed", "pi0_1_transformed", "pi0_2_transformed", 
                           "rho_1_transformed", "rho_2_transformed", "aco_angle_1", "y_1_1", "y_1_2", 
                           "m_1", "m_2", "w_a", "w_b", "E_miss", "E_miss_x", "E_miss_y", 
                           "aco_angle_5", "aco_angle_6", "aco_angle_7",
                           "y"]
        self.layers = 0
        self.config_num = 0
        self.binary = binary
    
    def createTrainTestData(self, save=True):
        if self.binary:
            y_sm = pd.DataFrame(np.ones(self.df_rho_sm.shape[0]))
            y_ps = pd.DataFrame(np.zeros(self.df_rho_ps.shape[0]))
            self.y = pd.concat([y_sm, y_ps]).to_numpy()
            df = pd.concat([self.df_rho_sm, self.df_rho_ps]).drop(["wt_cp_sm","wt_cp_ps","wt_cp_mm", "rand", 
                "tau_decay_mode_1","tau_decay_mode_2","mva_dm_1","mva_dm_2",], axis=1).reset_index(drop=True)
        else:
            df = self.df_rho_clean
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        # boost into rest frame of resonances
        rest_frame = pi_1 + pi_2 + pi0_1 + pi0_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi0_2_boosted = pi0_2.boost_particle(boost)
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
            rot_mat = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0,0,1])
            pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
            pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
            pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
            pi0_2_boosted_rot.append(rot_mat.dot(pi0_2_boosted[1:, i]))
            rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
            rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
            if i%100000==0:
                print('finished getting rotated 4-vector', i)
        self.pi_1_transformed = np.c_[pi_1_boosted[0], np.array(pi_1_boosted_rot)]
        self.pi_2_transformed = np.c_[pi_2_boosted[0], np.array(pi_2_boosted_rot)]
        self.pi0_1_transformed = np.c_[pi0_1_boosted[0], np.array(pi0_1_boosted_rot)]
        self.pi0_2_transformed = np.c_[pi0_2_boosted[0], np.array(pi0_2_boosted_rot)]
        self.rho_1_transformed = np.c_[rho_1_boosted[0], np.array(rho_1_boosted_rot)]
        self.rho_2_transformed = np.c_[rho_2_boosted[0], np.array(rho_2_boosted_rot)]
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
        # read met_data
        self.E_miss = df['met']
        self.E_miss_x = df['metx']
        self.E_miss_y = df['mety']
        if save:
            print('Saving train/test data to file')
            to_save = [self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, 
                       self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, 
                       self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y,
                       self.aco_angle_5, self.aco_angle_6, self.aco_angle_7]
            if self.binary:
                to_save += [self.y]
            for i in range(len(to_save)):
                if self.binary:
                    save_name = f'{self.save_dir}/{self.file_names[i]}_b'
                else:
                    save_name = f'{self.save_dir}/{self.file_names[i]}'
                np.save(save_name, to_save[i], allow_pickle=True)
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

    def chooseConfigMap(self):
        # TODO: choose config map once
        pass

    def configTrainTestData(self, config_num):
        self.config_num = config_num
        config_map = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1 , self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            7: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
        }
        config_map_norho = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1 , self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
        }
        config_map_onlyrho = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.rho_1_transformed, self.rho_2_transformed],
            4: np.c_[self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1 , self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
        }
        try:
            # self.X = config_map[self.config_num]
            self.X = config_map_norho[self.config_num]
            # self.X = config_map_onlyrho[self.config_num]
        except KeyError as e:
            print('Wrong config input number')
            exit(-1)
        if self.binary:
            # self.y.astype(int) # probably doesn't matter
            # self.y = (~(self.df_rho["rand"]<self.df_rho["wt_cp_ps"]/2).to_numpy()).astype(int) #kristof's method
            self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X, self.y, test_size=0.2, random_state=123456, stratify=self.y,)
        else:
            self.y = (self.w_a/(self.w_a+self.w_b))
            self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X, self.y, test_size=0.2, random_state=123456,)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
   
    def createConfigStr(self):
        if self.binary:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}_binary'
        else:
            config_str = f'config{self.config_num}_{self.layers}_{self.epochs}_{self.batch_size}'
        return config_str

    def trainRepeated(self, repeated_num, external_model, epochs, batch_size, patience=10):
        auc_score_arr = []
        file = f'{self.write_dir}/{self.write_filename}_repeats.txt'
        for i in range(repeated_num):
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training {i} out of {repeated_num} times~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            self.train(external_model=external_model, epochs=epochs, batch_size=batch_size, patience=patience)
            auc_score = self.evaluation(write=False)
            auc_score_arr.append(auc_score)
        auc_avg = np.average(auc_score_arr)
        auc_error = np.std(auc_score_arr, ddof=1)
        self.writeMessage(file, f'{auc_avg},{auc_error},{self.config_num},{self.layers},{self.epochs},{self.batch_size},{self.binary}')
        

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
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
        self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        callbacks=[self.history,early_stop],
                        validation_data=(self.X_test, self.y_test))
        # self.model.save(f'{self.save_dir}/NN_1')

    def evaluation(self, write=True):
        # use test dataset for evaluation
        if self.binary:
            y_proba = self.model.predict_proba(self.X_test) # outputs two probabilties
            # print(y_proba)
            auc = roc_auc_score(self.y_test, y_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            self.plot_roc_curve(fpr, tpr, auc)
        else:
            y_pred_test = self.model.predict(self.X_test)
            _, w_a_test, _, w_b_test = train_test_split(self.w_a, self.w_b, test_size=0.2, random_state=123456)
            auc, y_label_roc, y_pred_roc = self.custom_auc_score(y_pred_test, w_a_test, w_b_test)
            fpr, tpr, _ = roc_curve(y_label_roc, y_pred_roc, sample_weight=np.r_[w_a_test, w_b_test])
            self.plot_roc_curve(fpr, tpr, auc)
        
        if write:
            file = f'{self.write_dir}/{self.write_filename}.txt'
            with open(file, 'a+') as f:
                print('Writing to file')
                f.write(f'{auc},{self.config_num},{self.layers},{self.epochs},{self.batch_size},{self.binary}\n')
            print('Finish writing')
            f.close()
        return auc

    def custom_auc_score(self, pred, w_a, w_b):
        set_a = np.ones(len(pred))
        set_b = np.zeros(len(pred))
        y_pred_roc = np.r_[pred, pred]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[w_a, w_b]
        custom_auc = roc_auc_score(y_label_roc, y_pred_roc, sample_weight=w_roc)
        return custom_auc, y_label_roc, y_pred_roc

    def func_model(self, units=[300,300,300], activations=['relu','relu','relu'], batch_norm=False, dropout=False):
        # TODO: make batch normalisation configurations and dropout, paper uses batch normalisation before activation
        if len(units) != len(activations):
            print('units and activations have different lengths')
            exit(-1)
        self.layers = len(units)
        inputs = tf.keras.Input(shape=(self.X.shape[1],))
        x = inputs
        for i in range(self.layers):
            x = tf.keras.layers.Dense(units[i], activation=None, kernel_initializer='normal')(x)
            if batch_norm:
                x = tf.keras.layers.BatchNormalization()(x)
            if dropout:
                x = tf.keras.layers.Dropout(0.2)(x)
            x = tf.keras.layers.Activation(activations[i])(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        # print(self.model.summary())
        return self.model

    def seq_model(self, units=[300,300], batch_norm=False, dropout=None):
        self.model = tf.keras.models.Sequential()
        for unit in units:
            self.model.add(tf.keras.layers.Dense(unit, kernel_initializer='normal'))
            if batch_norm:
                self.model.add(tf.keras.layers.normalization.BatchNormalization())
            self.model.add(tf.keras.layers.Activation('relu'))
            if dropout is not None:
                self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')  
        return self.model

    def kristof_model(self, dimensions=-1):
        # model by kristof
        if dimensions == -1:
            dimensions = self.X.shape[1]
        # create model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(38, input_dim=dimensions, kernel_initializer='normal', activation='relu'))
        self.model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')  
        return self.model

    def writeMessage(self, file, message):
        with open(file, 'a+') as f:
            print(f'Writing {message} to file')
            f.write(f'{message}\n')
        print('Finish writing')
        f.close()

    def setModel(self, model):
        self.model = model

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
        
    def aucTest(self):
        # TODO: change w_roc
        # theroetical limit (non-binary labels)
        set_a = np.ones(len(self.y))
        set_b = np.zeros(len(self.y))
        y_pred_roc = np.r_[self.y, self.y]
        y_label_roc = np.r_[set_a, set_b]
        w_roc = np.r_[self.w_a, self.w_b]
        custom_auc = roc_auc_score(y_label_roc, y_pred_roc, sample_weight=w_roc)
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

if __name__ == '__main__':
    # set up NN
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Setting up NN~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    NN = Potential2016(binary=True)
    read = False
    if not read:
        NN.readData()
        NN.cleanData()
        NN.createTrainTestData()
    else:
        NN.readTrainTestData()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished NN setup~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # NN.setModel(NN.seq_model())
    # NN.configTrainTestData(3)
    # NN.train(external_model=True, epochs=20, batch_size=2048)
    # # NN.aucTest()
    # NN.evaluation(write=True)
    # NN.plotLoss()
    NN_config = [
        [[300,300], False, False],
        # [[300,300], True, False],
        # [[300,300], False, True],
        # [[300,300], True, True],
    ]
    file = 'potential_'
    
    # for config in NN_config:
    #     NN.setModel(NN.seq_model(*config))
    #     # NN.writeMessage('')
    #     for i in range(1, 7):
    #         print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #         NN.configTrainTestData(i)
    #         NN.trainRepeated(5, external_model=True, epochs=20, batch_size=2048)

    for i in range(1, 7):
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Training config {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # NN.writeMessage('')
        NN.configTrainTestData(i)
        NN.setModel(NN.seq_model())
        NN.train(external_model=True, epochs=20, batch_size=2048)
        NN.evaluation(write=True)
        NN.plotLoss()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')