"""
Neural net to regress phi star from the lambdas.
Includes neural nets for regressing O star and the lambdas as well.
"""

import uproot 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
#from tensorflow.keras.layers import BatchNormalization
#from lbn_modified_Alie import LBN, LBNLayer
#from lbn import LBN, LBNLayer
#from datetime import datetime
#from ROOT import TLorentzVector
from pylorentz import Momentum4


class NN_phi_star():
    
    def __init__(self):
        self.want_model_image = True
        self.NN_name = 'phi_star'
    
    def readData(self):
        self.tree_tt = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_tt.root")["ntuple"]
        #self.tree_et = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_et.root")["ntuple"]
        #self.tree_mt = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_mt.root")["ntuple"]
    
    def cleanData(self):
        variables = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1", 
            "mva_dm_1","mva_dm_2",
            "tau_decay_mode_1","tau_decay_mode_2",
        #     "ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2", # ignore impact parameter for now
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", 
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", 
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
            "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", 
            "y_1_1", "y_1_2"
        ]
        self.df = self.tree_tt.pandas.df(variables)
        # select only rho-rho events
        self.df_rho = self.df[(self.df['mva_dm_1']==1) & (self.df['mva_dm_2']==1) & (self.df["tau_decay_mode_1"] == 1) & (self.df["tau_decay_mode_2"] == 1)]
        # select ps and sm data
        self.df_rho_ps = self.df_rho[(self.df_rho["rand"]<self.df_rho["wt_cp_ps"]/2)]
        self.df_rho_sm = self.df_rho[(self.df_rho["rand"]<self.df_rho["wt_cp_sm"]/2)]
        # drop unnecessary labels 
        self.df_rho_clean = self.df_rho.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        self.df_rho_ps_clean = self.df_rho_ps.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        self.df_rho_sm_clean = self.df_rho_sm.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def parseData(self, df):
        #X = df.drop(["y_1_1", "y_1_2", "aco_angle_1"], axis=1).reset_index(drop=True)
        #y = df[['aco_angle_1']]
        #X = np.reshape(X.to_numpy(), (X.shape[0], 4, 4))
        #y = y.to_numpy()
        
        # create 4-vectors
        p3 = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        p4 = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        p1 = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        p2 = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        
        # boost particles
        zmf = p3 + p4
        p3_zmf = p3.boost_particle(-zmf)
        p4_zmf = p4.boost_particle(-zmf)
        p1_zmf = p1.boost_particle(-zmf)
        p2_zmf = p2.boost_particle(-zmf)
        
        # calculate the lambdas
        p1_trans = np.cross(p1_zmf[1:,:].transpose(), p3_zmf[1:, :].transpose())
        p2_trans = np.cross(p2_zmf[1:,:].transpose(), p4_zmf[1:, :].transpose())

        # normalize lambdas (n1, n2)
        n1 = p1_trans/np.linalg.norm(p1_trans, ord=2, axis=1, keepdims=True)
        n2 = p2_trans/np.linalg.norm(p2_trans, ord=2, axis=1, keepdims=True)
        
        #Calculate Phi_ZMF using dot product and arccos
        phi_star = np.arccos(np.sum(n1*n2, axis=1))
        
        X = np.concatenate([n1, n2], axis=1) # for a shape of (?, 6)
        #X = np.stack([n1, n2], axis=1) # for a shape of (?, 2, 3)
        y = phi_star
        
        return X, y

    def createTrainTestData(self):
        self.X, self.y = self.parseData(self.df_rho_clean)
        self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X, self.y, test_size=0.2, random_state=123456)

    def train(self, loss_function='mean_squared_error', epochs=50, batch_size=1000, patience=10):
        self.epochs = epochs
        self.batch_size = batch_size
        if isinstance(loss_function, str):
            self.loss_function = loss_function
        else:
            self.loss_function = loss_function.__name__
        self.model = self.baseline_model(loss_function)
        
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.history,early_stop], validation_data=(self.X_test, self.y_test))
        
    def baseline_model(self, loss_fn):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(30, input_dim=6, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=loss_fn, optimizer='adam')  
        return model

    def plotLoss(self):
        plt.clf()
        # Extract number of run epochs from the training history
        epochs = range(1, len(self.history.history["loss"])+1)
        # Extract loss on training and validation ddataset and plot them together
        plt.plot(epochs, self.history.history["loss"], "o-", label="Training")
        plt.plot(epochs, self.history.history["val_loss"], "o-", label="Test")
        plt.xlabel("Epochs"), plt.ylabel("Loss")
        # plt.yscale("log")
        plt.legend()
        plt.savefig(f'./task2/{self.NN_name}_loss_{self.epochs}_{self.batch_size}_{self.loss_function}')
        plt.show()
    
    def evaluation(self, write=True, verbose=False):
        if self.NN_name == 'phi_star':
            prediction = self.model.predict(self.X_test).flatten()
        else:
            prediction = self.model.predict(self.X_test)
        # df_y = pd.DataFrame(self.y_test, columns=['aco_angle_1'])
        # df_y['prediction'] = prediction
        print('self.y_test.shape =', self.y_test.shape)
        print('prediction.shape =', prediction.shape)
        self.model.summary()
        if self.want_model_image:
            tf.keras.utils.plot_model(self.model, "./task2/model.png", show_shapes=True)
        self.test_loss = self.model.evaluate(self.X_test, self.y_test)
        self.r_2_score = r2_score(self.y_test, prediction)
        self.mae = mean_absolute_error(self.y_test, prediction)
        self.mse = mean_squared_error(self.y_test, prediction)
        if verbose:
            print(f'Test loss: {self.test_loss}')
            print(f"R**2 score is: {self.r_2_score}")
            print(f"MAE is: {self.mae}")
            print(f"MSE is: {self.mse}")
        if write:
            file = './Task_2_' + self.NN_name + '.txt'
            with open(file, 'a+') as f:
                print('Writing to file')
                f.write(f'{self.loss_function},{self.epochs},{self.batch_size},{self.test_loss},{self.r_2_score},{self.mae},{self.mse}\n')
            print('Finish writing')
            f.close()

    def plotDistribution(self, bins=100, sample_num=5000):
        plt.clf()
        X_ps, y_ps = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        X_sm , y_sm = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        ps = self.model.predict(X_ps).flatten()
        sm = self.model.predict(X_sm).flatten()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
        ax1.hist(ps, bins=bins, alpha=0.5)
        ax1.hist(sm, bins=bins, alpha=0.5)
        ax1.set_ylabel('freq')
        ax1.set_xlabel(self.NN_name)
        ax1.set_title('regression using NN')
        ax2.hist(y_ps, bins=bins, alpha=0.5)
        ax2.hist(y_sm, bins=bins, alpha=0.5)
        ax2.set_ylabel('freq')
        ax2.set_xlabel(self.NN_name)
        ax2.set_title('true values')
        plt.tight_layout()
        plt.savefig(f"./task2/{self.NN_name}_angledist_{self.epochs}_{self.batch_size}_{self.loss_function}")


        
class NN_lambdas(NN_phi_star):
    
    def __init__(self):
        super().__init__()
        self.NN_name = 'lambdas'

    def parseData(self, df):
        #X = df.drop(["y_1_1", "y_1_2", "aco_angle_1"], axis=1).reset_index(drop=True)
        #y = df[['aco_angle_1']]
        #X = np.reshape(X.to_numpy(), (X.shape[0], 4, 4))
        #y = y.to_numpy()
        
        # create 4-vectors
        p3 = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        p4 = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        p1 = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        p2 = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        
        # boost particles
        zmf = p3 + p4
        p3_zmf = p3.boost_particle(-zmf)
        p4_zmf = p4.boost_particle(-zmf)
        p1_zmf = p1.boost_particle(-zmf)
        p2_zmf = p2.boost_particle(-zmf)
        
        # calculate the lambdas
        p1_trans = np.cross(p1_zmf[1:,:].transpose(), p3_zmf[1:, :].transpose())
        p2_trans = np.cross(p2_zmf[1:,:].transpose(), p4_zmf[1:, :].transpose())

        # normalize lambdas (n1, n2)
        n1 = p1_trans/np.linalg.norm(p1_trans, ord=2, axis=1, keepdims=True)
        n2 = p2_trans/np.linalg.norm(p2_trans, ord=2, axis=1, keepdims=True)
        
        X = np.concatenate([np.array(p1_zmf).transpose(), np.array(p2_zmf).transpose(), np.array(p3_zmf).transpose(), np.array(p4_zmf).transpose()], axis=1)
        y = np.concatenate([n1, n2], axis=1) # for a shape of (?, 6)
        
        return X, y
    
    def baseline_model(self, loss_fn):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(30, input_dim=16, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(6))
        model.compile(loss=loss_fn, optimizer='adam')  
        return model

    def plotDistribution(self, bins=100, sample_num=5000):
        plt.clf()
        X_ps, y_ps = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        X_sm , y_sm = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        y_ps = y_ps[:, 0]
        y_sm = y_sm[:, 0]
        ps = self.model.predict(X_ps)[:, 0]
        sm = self.model.predict(X_sm)[:, 0]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))
        ax1.hist(ps, bins=bins, alpha=0.5)
        ax1.hist(sm, bins=bins, alpha=0.5)
        ax1.set_ylabel('freq')
        ax1.set_xlabel('lambda + first component')
        ax1.set_title('regression using NN')
        ax2.hist(y_ps, bins=bins, alpha=0.5)
        ax2.hist(y_sm, bins=bins, alpha=0.5)
        ax2.set_ylabel('freq')
        ax2.set_xlabel('lambda + first component')
        ax2.set_title('true values')
        plt.tight_layout()
        plt.savefig(f"./task2/{self.NN_name}_angledist_{self.epochs}_{self.batch_size}_{self.loss_function}")

    
    
class NN_O_star(NN_phi_star):
    
    def __init__(self):
        super().__init__()
        self.NN_name = 'O_star'

    def parseData(self, df):
        #X = df.drop(["y_1_1", "y_1_2", "aco_angle_1"], axis=1).reset_index(drop=True)
        #y = df[['aco_angle_1']]
        #X = np.reshape(X.to_numpy(), (X.shape[0], 4, 4))
        #y = y.to_numpy()
        
        # create 4-vectors
        p3 = Momentum4(df["pi_E_1"], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        p4 = Momentum4(df["pi_E_2"], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        p1 = Momentum4(df["pi0_E_1"], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        p2 = Momentum4(df["pi0_E_2"], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        
        # boost particles
        zmf = p3 + p4
        p3_zmf = p3.boost_particle(-zmf)
        p4_zmf = p4.boost_particle(-zmf)
        p1_zmf = p1.boost_particle(-zmf)
        p2_zmf = p2.boost_particle(-zmf)
        
        # calculate the lambdas
        p1_trans = np.cross(p1_zmf[1:,:].transpose(), p3_zmf[1:, :].transpose())
        p2_trans = np.cross(p2_zmf[1:,:].transpose(), p4_zmf[1:, :].transpose())

        # normalize lambdas (n1, n2)
        n1 = p1_trans/np.linalg.norm(p1_trans, ord=2, axis=1, keepdims=True)
        n2 = p2_trans/np.linalg.norm(p2_trans, ord=2, axis=1, keepdims=True)

        #calculate O_star
        O_star = np.sum(np.cross(n1, n2).transpose()*np.array(p4_zmf[1:, :]), axis=0)
        
        X = np.concatenate([np.array(p4_zmf).transpose(), n1, n2], axis=1)
        y = np.array(O_star)
        #print(X.shape)
        #print(y.shape)
        
        return X, y

    def baseline_model(self, loss_fn):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(30, input_dim=10, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(100, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss=loss_fn, optimizer='adam')  
        return model
    
    
        
def main():
    
    #NN = NN_phi_star()
    #NN = NN_lambdas()
    NN = NN_O_star()
    
    NN.readData()
    print('Reading done')
    NN.cleanData()
    NN.createTrainTestData()
    print('Train test split done')
    NN.train(epochs=10, batch_size=1000)
    print('Training done')
    NN.plotLoss()
    NN.evaluation(write=True, verbose=True)
    NN.plotDistribution()
    
if __name__=='__main__':
    main()
    