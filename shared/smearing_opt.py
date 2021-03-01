from data_loader import DataLoader
from config_loader import ConfigLoader
from config_checker import ConfigChecker
from tuner import Tuner
from smearer import Smearer
from sklearn.metrics import roc_auc_score
import os
import hyperopt
import tensorflow as tf
import numpy as np
import config
import random
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import config
import uproot
import pandas as pd

seed_value = config.seed_value
# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set the `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.compat.v1.set_random_seed(seed_value)



class SmearingOptimizer:
    def __init__(self, channel):
        """
        binary is always TRUE
        """
        self.addons_config_gen = {'neutrino': {'load_alpha':False, 'termination':1000, 'imputer_mode':'remove', 'save_alpha':True,}, 'met': {}, 'ip':{}, 'sv': {}}
        self.channel = channel


    def run(self, from_hdf=True):
        self.epochs, self.batch_size = self.setModel()
        self.initializeWithSmear(self.addons_config_gen, from_hdf=from_hdf)
        # read smearing root file
        print(f'(SmearerOpt) Loading .root info with using HDF5 as {from_hdf}')
        df_gen_reco = self.readSmearingData(from_hdf=from_hdf)
        print('(SmearerOpt) Cleaning data')
        self.df_gen_reco_clean = self.cleanSmearingData(df_gen_reco)
        if self.channel == 'rho_rho':
            # possible_features = ['pi_1', 'pi0_1', 'pi_2', 'pi0_2', 'metx', 'mety', 'ip_1', 'ip_2', 'sv_1', 'sv_2']
            possible_features = ['pi_1', 'pi0_1']
        elif self.channel == 'rho_a1':
            possible_features = ['pi_1', 'pi0_1', 'pi_2', 'pi2_2', 'pi3_2', 'metx', 'mety', 'ip_1', 'ip_2', 'sv_1', 'sv_2']
        else:
            possible_features = ['pi_1', 'pi2_1', 'pi3_1', 'pi_2', 'pi2_2', 'pi3_2', 'metx', 'mety', 'ip_1', 'ip_2', 'sv_1', 'sv_2']

        space = {
                'features': hp.choice('features', possible_features),
            }
        trials = Trials()
        # best = fmin(self.smearingObj, space, algo=tpe.suggest, trials=trials, max_evals=int(np.ceil(len(possible_features))))
        best = fmin(self.smearingObj, space, algo=tpe.suggest, trials=trials, max_evals=2) 
        best_params = hyperopt.space_eval(space, best)
        print(best_params)
        print(trials.trials)
        # params = {
        #     'features': ['pi_1'],
        #     'epochs': epochs,
        #     'batch_size': batch_size,
        # }
        # self.smearingObj(params)
        return trials, best

    def smearingObj(self, params):
        features = params['features']
        print(f'On feature: {features}')
        # print(type(features))
        df_smeared_transformed = self.smearAndCreateInputData([features])
        CL = ConfigLoader(df_smeared_transformed, self.channel, True)
        X_train, X_test, y_train, y_test = CL.configTrainTestData(self.config_num, True)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=20)
        self.model.fit(X_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[early_stop],
                       validation_data=(X_test, y_test),
                       verbose=0)

        y_proba = self.model.predict(X_test)  # outputs two probabilties
        auc = roc_auc_score(y_test, y_proba)
        return {'loss': auc, 'status': STATUS_OK}


    def initializeWithSmear(self, addons_config={}, from_hdf=True):
        """ NO READ FUNCTION - ALWAYS CREATE SMEARING DIST """
        if self.channel == 'rho_rho':
            variables = config.variables_gen_rho_rho
        elif self.channel == 'rho_a1':
            variables = config.variables_gen_rho_a1
        elif self.channel == 'a1_a1':
            variables = config.variables_gen_a1_a1
        else:
            raise ValueError('Incorrect channel inputted')
        self.DL = DataLoader(variables, self.channel, True) 
        CC = ConfigChecker(self.channel, True, True)
        CC.checkInitialize(self.DL, addons_config, False, from_hdf)
        print(f'Loading .root info with using HDF5 as {from_hdf}')
        df_orig = self.DL.readGenData(from_hdf=from_hdf)
        print('Cleaning data')
        self.df_clean, _, _ = self.DL.cleanGenData(df_orig)
        

    def readSmearingData(self, from_hdf=False):
        if not from_hdf:
            tree_tt = uproot.open(Smearer.smearing_root_path)["ntuple"]
            df = tree_tt.pandas.df(self.variables)
            df.to_hdf(f"{Smearer.smearing_df_path}_{self.channel}.h5", 'df')
        else:
            df = pd.read_hdf(f"{Smearer.smearing_df_path}_{self.channel}.h5", 'df')
        return df

    def cleanSmearingData(self, df):
        """exactly the same as cleanGenData"""
        if self.channel == 'rho_rho':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 1)]
        elif self.channel == 'rho_a1':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 10)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['dm_1'] == 10) & (df['dm_2'] == 10)]
        else:
            raise ValueError('Incorrect channel inputted')
        df_clean = df_clean.dropna()
        df_clean = df_clean[(df_clean == -9999.0).sum(1) < 2]
        return df_clean


    def smearAndCreateInputData(self, feature, addons_config={}):
        if not addons_config:
            addons = []
        else:
            addons = addons_config.keys()
        if self.channel == 'rho_rho':
            variables_smear = config.variables_smearing_rho_rho
        elif self.channel == 'rho_a1':
            variables_smear = config.variables_smearing_rho_a1
        elif self.channel == 'a1_a1':
            variables_smear = config.variables_smearing_a1_a1
        SM = Smearer(variables_smear, self.channel, feature)
        df_smeared = self.df_clean.copy()
        # print('(Smearer) Creating smearing distribution')
        for base_feature in SM.features_to_smear:
            SM.createSmearedDataForOneBaseFeature(self.df_gen_reco_clean, df_smeared, base_feature, SM.features_to_smear[base_feature], plot=False)
        # print(f'Smeared: {self.features_to_smear}')
        # remove Nans
        df_smeared = df_smeared.dropna()
        df_ps_smeared, df_sm_smeared = SM.selectPSSMFromData(df_smeared)
        df_smeared_transformed = self.DL.createTrainTestData(df_smeared, df_ps_smeared, df_sm_smeared, True, True, addons, addons_config, save=False)
        # df_orig_transformed = self.DL.createTrainTestData(df_clean, df_ps_clean, df_sm_clean, binary, True, addons, addons_config, save=False)
        # deal with imaginary numbers from boosting
        # df_smeared_transformed = df_smeared_transformed.apply(np.real)
        # m_features = [x for x in df_smeared_transformed.columns if x.startswith('m')]
        # for m_feature in m_features:
        #     df_smeared_transformed = df_smeared_transformed[df_smeared_transformed[m_feature]!=0]
        # debugging part
        # df.to_hdf('smearing/df_smeared.h5', 'df')
        # df_smeared.to_hdf('./smearing/df_smeared.h5', 'df')
        # df_clean.to_hdf('./smearing/df_orig.h5', 'df')
        # df_smeared_transformed.to_hdf('smearing/df_smeared_transformed.h5', 'df')
        # df_orig_transformed.to_hdf('smearing/df_orig_transformed.h5', 'df')
        # exit()
        return df_smeared_transformed

    def setModel(self):
        """
        Sets self.model and self.config_num
        """
        with open('./NN_output/smearing_hp.txt', 'r') as fh:
            num_list = [line for line in fh]
        if self.channel == 'rho_rho':
            self.config_num = 1.6
            nn_arch = num_list[0].split(',')
        elif self.channel == 'rho_a1':
            self.config_num = 1.6
            nn_arch = num_list[1].split(',')
        else:
            self.config_num = 1.6
            nn_arch = num_list[2].split(',')
        optimal_auc = float(nn_arch[1])
        nodes = int(nn_arch[3])
        num_layers = int(nn_arch[4])
        batch_norm = bool(nn_arch[5])
        dropout = float(nn_arch[6])
        epochs = int(nn_arch[7])
        batch_size = int(nn_arch[8])
        learning_rate = float(nn_arch[11])
        activation = str(nn_arch[12])
        initializer_std = float(nn_arch[13])
        params = {
            'nodes': nodes,
            'num_layers': num_layers,
            'batch_norm': batch_norm,
            'dropout': dropout,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'activation': activation,
            'initializer_std': initializer_std,
        }
        print(f'Training with {params}')
        tuner = Tuner()
        self.model, _ = tuner.hyperOptModelNN(params)
        return epochs, batch_size

if __name__ == '__main__':
    SmOpt = SmearingOptimizer('rho_rho')
    trials, best = SmOpt.run(from_hdf=True)