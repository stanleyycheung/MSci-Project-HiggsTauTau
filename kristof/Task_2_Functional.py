import uproot 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from lbn import LBN, LBNLayer


class NN_aco_angle_1:
    
    def __init__(self):
        self.simpler_shape = False #this is true if we want the data in shape of (16,). If false, then want shape (4,4)
        
    def readData(self):
        self.tree_tt = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_tt.root")["ntuple"]
        self.tree_et = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_et.root")["ntuple"]
        self.tree_mt = uproot.open("/eos/user/k/kgalambo/SWAN_projects/Masters_CP_Kristof_2/MVAFILE_AllHiggs_mt.root")["ntuple"]
    
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
        X = df.drop(["y_1_1", "y_1_2", "aco_angle_1"], axis=1).reset_index(drop=True)
        y = df[['aco_angle_1']]
        X = np.reshape(X.to_numpy(), (X.shape[0], 4, 4))
        y = y.to_numpy()
        return X, y

    def createTrainTestData(self):
        self.X, self.y = self.parseData(self.df_rho_clean)
        self.X_train, self.X_test, self.y_train, self.y_test  = train_test_split(self.X, self.y, test_size=0.2, random_state=123456)
    
    def train(self, loss_function='mean_squared_error', epochs=50, batch_size=1000, patience=10, mode='sequential'):
        self.epochs = epochs
        self.batch_size = batch_size
        if isinstance(loss_function, str):
            self.loss_function = loss_function
        else:
            self.loss_function = loss_function.__name__
        if mode == 'sequential':
            self.model = self.lbn_model(loss_function)
        elif mode == 'functional_kingsley':
            self.model = self.functional_model_kingsley(loss_function)
        elif mode == 'functional_basic':
            self.model = self.functional_model_basic(loss_function)
            self.simpler_shape = True
        elif mode == 'functional':
            self.model = self.functional_model(loss_function)
            #self.simpler_shape = True
        elif mode == 'functional_advanced':
            self.model = self.functional_model_advanced(loss_function)
            #self.simpler_shape = True
        else:
            raise Exception('mode not understood!!!')
        if self.simpler_shape:
            self.X_train = np.reshape(self.X_train, (-1, 16))
            self.X_test = np.reshape(self.X_test, (-1, 16))
        self.history = tf.keras.callbacks.History()
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)

        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=[self.history,early_stop], validation_data=(self.X_test, self.y_test))
    
    def run(self, loss_function, epochs=50, batch_size=1000):
        # TO CHANGE
        self.readData()
        self.cleanData()
        self.createTrainTestData()
        self.train(loss_function, epochs, batch_size)
    
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
        plt.savefig(f'./task2/loss_{self.epochs}_{self.batch_size}_{self.loss_function}')
        plt.show()
    
    def evaluation(self, write=True, verbose=False):
        prediction = self.model.predict(self.X_test).flatten()
        # df_y = pd.DataFrame(self.y_test, columns=['aco_angle_1'])
        # df_y['prediction'] = prediction
        print('self.y_test.shape =', self.y_test.shape)
        print('prediction.shape =', prediction.shape)
        self.model.summary()
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
            file = './Task_2.txt'
            with open(file, 'a+') as f:
                print('Writing to file')
                f.write(f'{self.loss_function},{self.epochs},{self.batch_size},{self.test_loss},{self.r_2_score},{self.mae},{self.mse}\n')
            print('Finish writing')
            f.close()

    def plotDistribution(self, bins=100, sample_num=5000):
        plt.clf()
        X_ps, y_ps = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        X_sm , y_sm = self.parseData(self.df_rho_ps_clean.sample(sample_num))
        if self.simpler_shape:
            X_ps = np.reshape(X_ps, (-1, 16))
            X_sm = np.reshape(X_sm, (-1, 16))
        ps = self.model.predict(X_ps).flatten()
        sm = self.model.predict(X_sm).flatten()
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))
        ax1.hist(ps, bins=bins, alpha=0.5)
        ax1.hist(sm, bins=bins, alpha=0.5)
        ax1.set_ylabel('freq')
        ax1.set_xlabel(r'$\phi_{CP}$')
        ax1.set_title('combined ps and sm distribution')
        ax2.hist(ps, bins=bins, color='tab:blue')
        ax2.set_title('ps distribution')
        ax2.set_ylabel('freq')
        ax2.set_xlabel(r'$\phi_{CP}$')
        ax3.hist(sm, bins=bins, color='tab:orange')
        ax3.set_title('sm distribution')
        ax3.set_ylabel('freq')
        ax3.set_xlabel(r'$\phi_{CP}$')
        plt.tight_layout()
        plt.savefig(f"./task2/angledist_{self.epochs}_{self.batch_size}_{self.loss_function}")

    def functional_model_kingsley(self, loss_fn):
        tf_zmf_momenta = tf.keras.Input(shape=(4,4))
        
        #Can I combine these 2 somehow?:
        tf_pi0_1_trans = tf.math.l2_normalize(tf.linalg.cross(tf_zmf_momenta[:,2,1:], tf_zmf_momenta[:,0,1:]), axis=1)
        tf_pi0_2_trans = tf.math.l2_normalize(tf.linalg.cross(tf_zmf_momenta[:,3,1:], tf_zmf_momenta[:,1,1:]), axis=1)
        tf_phi_shift_0 = tf.math.acos(tf.reduce_sum(tf.math.multiply(tf_pi0_1_trans, tf_pi0_2_trans), axis=1))
        tf_big_O = tf.math.reduce_sum(tf.math.multiply(tf.linalg.cross(tf_pi0_1_trans, tf_pi0_2_trans), tf_zmf_momenta[:,1,1:]), axis=1)
        tf_phi_shift_1 = tf.where(tf_big_O<0, tf_phi_shift_0, 2*np.pi-tf_phi_shift_0)
        tf_y_1 = (tf_zmf_momenta[:,0,0] - tf_zmf_momenta[:,2,0])/(tf_zmf_momenta[:,0,0] + tf_zmf_momenta[:,2,0])
        yf_y_2 = (tf_zmf_momenta[:,1,0] - tf_zmf_momenta[:,3,0])/(tf_zmf_momenta[:,1,0] + tf_zmf_momenta[:,3,0])
        tf_y_tau = tf_y_1*yf_y_2
        tf_phi_shift_2 = tf.where(tf_y_tau<0, tf_phi_shift_1, tf.where(tf_phi_shift_1<np.pi, tf_phi_shift_1+np.pi, tf_phi_shift_1-np.pi))
        
        model = tf.keras.Model(inputs=tf_zmf_momenta, outputs=tf_phi_shift_2)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
        return model
        
    def functional_model_basic(self, loss_fn):
        inputs = tf.keras.Input(shape=(16,))
        x = tf.keras.layers.Dense(64, activation="relu")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
        return model
    
    def functional_model(self, loss_fn):
        inputs = tf.keras.Input(shape=(4,4))
        input_shape = (4, 4)
        LBN_output_features = ["E", "px", "py", "pz", "m", "pair_dy", "pair_cos"]
        x = LBNLayer(input_shape, n_particles=4, boost_mode=LBN.PAIRS, features=LBN_output_features)(inputs)
        x = tf.keras.layers.Dense(300, activation="relu")(x)
        x = tf.keras.layers.Dense(300, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
        return model
    
    def functional_model_advanced(self, loss_fn):
        #first lbn layer does the lorentz boost
        inputs = tf.keras.Input(shape=(4, 4))
        input_shape = (4, 4)
        LBN_output_features = ["E", "px", "py", "pz"]
        x = LBNLayer(input_shape, n_particles=4, boost_mode=LBN.PAIRS, features=LBN_output_features)(inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        boosted = tf.keras.layers.Dense(16)(x)
        
        model1 = tf.keras.Model(inputs=inputs, outputs=boosted)
        #model1.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
        
        #second layer does the cross-product
        LBN_output_features = ["E", "px", "py", "pz", "m", "pair_dy", "pair_cos"] #should have a cross-product feature in here
        x = tf.keras.layers.Reshape((4, 4))(boosted)
        x = LBNLayer((4, 4), n_particles=4, boost_mode=LBN.PAIRS, features=LBN_output_features, name='LBN2')(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        lambdas = tf.keras.layers.Dense(1)(x)
        
        model2 = tf.keras.Model(inputs=inputs, outputs=lambdas)
        model2.compile(optimizer='adam', loss=loss_fn, metrics=['mae'])
        return model2
        
    def lbn_model(self, loss_fn):
        input_shape = (4, 4)
        LBN_output_features = ["E", "px", "py", "pz", "m", "pair_dy", "pair_cos"]
        model = tf.keras.models.Sequential()
        model.add(LBNLayer(input_shape, n_particles=4, boost_mode=LBN.PAIRS, features=LBN_output_features))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Dense(300, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(300, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        if not loss_fn:
            loss_fn = 'mean_squared_error'
        model.compile(optimizer='adam', loss=loss_fn)  
        # model.compile(optimizer='adam', loss=custom_mse)
        return model

    def baseline_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.models.Dense(len(self.X.columns), input_dim=len(self.X.columns), kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.models.Dense((len(self.X.columns))*2, kernel_initializer='normal', activation='relu'))
        model.add(tf.keras.models.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')  
        return model

def custom_mse_example(y_true, y_pred):
    # WORK IN PROGRESS
    rem_true = math.fmod(y_true, 2*math.pi)
    rem_pred = math.fmod(y_pred, 2*math.pi)
    if rem_true < 0: rem_true += 2*math.pi
    if rem_pred < 0: rem_pred += 2*math.pi
    dist = abs(rem_true - rem_pred)
    if dist > np.pi:
        dist = 2*math.pi - dist
    # print(rem_true, rem_pred, dist)
    return tf.math.reduce_mean(tf.square(dist))

def loss_0(y_true, y_pred):
    # WORK IN PROGRESS
    rem_true = tf.math.floormod(y_true, 2*math.pi)
    rem_pred = tf.math.floormod(y_pred, 2*math.pi)
    if rem_true < 0: rem_true = tf.math.add(rem_true, 2*math.pi)
    if rem_pred < 0: rem_pred += 2*math.pi

def loss_1(y_true, y_pred):
    # loss function of 2(1-cos(y_true - y_pred))
    return tf.math.reduce_mean(tf.multiply(tf.add(tf.constant([1], dtype=tf.float32), -tf.math.cos(y_true-y_pred)), 2))

def loss_2(y_true, y_pred):
    # loss function of sqrt(2(1-cos(y_true - y_pred)))
    return tf.math.sqrt(tf.math.reduce_mean(tf.multiply(tf.add(tf.constant([1], dtype=tf.float32), -tf.math.cos(y_true-y_pred)), 2)))

def loss_3(y_true, y_pred):
    # loss function of arctan(sin(y_true - y_pred)/cos(y_true - y_pred))
    return tf.math.reduce_mean(tf.math.abs(tf.math.atan2(tf.math.sin(y_true - y_pred), tf.math.cos(y_true - y_pred))))

def main():
    # set up NN
    NN = NN_aco_angle_1()
    NN.readData()
    NN.cleanData()
    NN.createTrainTestData()
    # MSE loss
    NN.train(epochs=50, batch_size=1000)
    NN.plotLoss()
    NN.evaluation(write=True, verbose=True)
    NN.plotDistribution()
    # custom loss fns
    loss_fns = [loss_1, loss_2, loss_3]
    for _ in loss_fns:
        NN.train(loss_function=_, epochs=50, batch_size=1000)
        NN.plotLoss()
        NN.evaluation(write=True, verbose=True)
        NN.plotDistribution()
    

    # loss_fn_list = [None, loss_1, loss3]
    # for loss_fn in loss_fn_list:
    #     NN.train(lbn_model(loss_fn), epochs=50, batch_size=1000)    

def main2():
    # set up NN
    NN = NN_aco_angle_1()
    NN.readData()
    print('Reading done')
    NN.cleanData()
    NN.createTrainTestData()
    print('Train test split done')
    
    NN.train(epochs=5, batch_size=1000, mode='functional_advanced')
    print('Training done')
    
    NN.plotLoss()
    NN.evaluation(write=True, verbose=True)
    NN.plotDistribution()
    
if __name__ == '__main__':
    #main()
    main2()
