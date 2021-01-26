import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from pylorentz import Momentum4

from neutrino_reconstructor import NeutrinoReconstructor


class DataLoader:
    """
    DataLoader class
    Functions:
    - Reads .root file (and saves df if necessary)
    - Selects appropirate channel, and distinguishes sm/ps events
    - Boosts, rotates events into correct rest frame, according to channel
    - Outputs df with all of the possible NN inputs

    To do:
    - Add in neutrino data

    Possible addons currently:
    - 'neutrino'
    - 'met'

    Note:
    - Hardcoded gen level variables
    """

    reco_root_path = "./MVAFILE_AllHiggs_tt.root"
    gen_root_path = "./MVAFILE_GEN_AllHiggs_tt.root"
    if os.path.exists("C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"):
        print('Using Kristof\'s .root file')
        reco_root_path = "C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"
    reco_df_path = './df_tt'
    gen_df_path = './df_tt_gen'
    input_df_save_dir = './input_df_reco'

    def __init__(self, variables, channel):
        """
        DataLoader should be near stateless, exceptions of the channel and variables needed to load
        Other instance variables should only deal with load/save directories
        """
        self.channel = channel
        self.variables = variables

    def loadRecoData(self, binary, addons=[]):
        """
        Loads the BR df directly from pickle - no need to read from .root, boost and rotate events
        """
        print('Reading reco df pkl file')
        addons_loaded = ""
        if addons:
            addons_loaded = '_'+'_'.join(addons)
        pickle_file_name = f'{DataLoader.input_df_save_dir}/input_{self.channel}{addons_loaded}'
        if binary:
            pickle_file_name += '_b'
        df_inputs = pd.read_pickle(pickle_file_name+'.pkl')
        return df_inputs

    def createRecoData(self, binary, from_pickle=True, addons=[], addons_config={}):
        """
        Creates the input (reco) data for the NN either from .root file or a previously saved .pkl file
        """
        print(f'Loading .root info with using pickle as {from_pickle}')
        df = self.readRecoData(from_pickle=from_pickle)
        print('Cleaning data')
        df_clean, df_ps_clean, df_sm_clean = self.cleanRecoData(df)
        print('Creating input data')
        df_inputs = self.createTrainTestData(df_clean, df_ps_clean, df_sm_clean, binary, addons, addons_config, save=True)
        return df_inputs

    def readRecoData(self, from_pickle=False):
        """
        Reads the reco root file, can save contents into .pkl for fast read/write abilities
        """
        if not from_pickle:
            tree_tt = uproot.open(DataLoader.reco_root_path)["ntuple"]
            # tree_et = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_et.root")["ntuple"]
            # tree_mt = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_mt.root")["ntuple"]
            df = tree_tt.pandas.df(self.variables)
            df.to_pickle(f"{DataLoader.reco_df_path}_{self.channel}.pkl")
        else:
            df = pd.read_pickle(f"{DataLoader.reco_df_path}_{self.channel}.pkl")
        return df

    def readGenData(self, from_pickle=False):
        """
        Reads the gen root file, can save contents into .pkl for fast read/write abilities
        Note: hardcoded variables for gen level 
        """
        variables_gen = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "dm_1", "dm_2",
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", # charged pion 1
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", # charged pion 2
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1", # neutral pion 1
            "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", # neutral pion 2,
            'metx', 'mety',
            'sv_x_1', 'sv_y_1', 'sv_z_1', 'sv_x_2', 'sv_y_2', 'sv_z_2',]
        if not from_pickle:
            tree_tt = uproot.open(DataLoader.gen_root_path)["ntuple"]
            df = tree_tt.pandas.df(variables_gen)
            df.to_pickle(f"{DataLoader.gen_df_path}_{self.channel}.pkl")
        else:
            df = pd.read_pickle(f"{DataLoader.gen_df_path}_{self.channel}.pkl")
        return df

    def cleanGenData(self, df):
        """
        Selects correct channel for gen data
        """
        if self.channel == 'rho_rho':
            df_clean = df[(df['dm_1']==1) & (df['dm_2']==1)]
        elif self.channel == 'rho_a1':
            df_clean = df[(df['dm_1']==1) & (df['dm_2']==10)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['dm_1']==10) & (df['dm_2']==10)]
        else:
            raise ValueError('Incorrect channel inputted')
        return df_clean

    def cleanRecoData(self, df):
        """
        Selects correct channel for reco data, whilst seperating sm/ps distributions as well
        """
        if self.channel == 'rho_rho':
            # select only rho-rho events
            df_clean = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df["tau_decay_mode_1"] == 1) & (df["tau_decay_mode_2"] == 1)]
            # select ps and sm data
            df_rho_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
            df_rho_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
            # drop unnecessary labels
            # df_clean = df_rho.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        elif self.channel == 'rho_a1':
            df_clean = df[(df['mva_dm_1']==1) & (df['mva_dm_2']==10) & (df["tau_decay_mode_1"] == 1)]
            df_rho_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
            df_rho_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['mva_dm_1']==10) & (df['mva_dm_2']==10)]
            df_rho_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
            df_rho_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
        else:
            raise ValueError('Incorrect channel inputted')
        return df_clean, df_rho_ps, df_rho_sm

    def augmentDfToBinary(self, df_ps, df_sm):
        y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
        y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))
        y = pd.concat([y_sm, y_ps]).to_numpy()
        df = pd.concat([df_sm, df_ps]).reset_index(drop=True)
        return df, y

    def createTrainTestData(self, df, df_ps, df_sm, binary, addons, addons_config, save=True):
        """
        Runs to create df with all NN input data, both test and train
        """
        if binary:
            print('In binary mode')
            # y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
            # y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))
            # y = pd.concat([y_sm, y_ps]).to_numpy()
            # df = pd.concat([df_sm, df_ps])
            df, y = self.augmentDfToBinary(df_ps, df_sm)
        else:
            y = None
        if self.channel == 'rho_rho':
            # df_inputs_data, boost = self.calculateRhoRhoData(df, len(df_ps))
            df_inputs_data, boost = self.calculateRhoRhoData_old(df)
        elif self.channel == 'rho_a1':
            df_inputs_data, boost = self.calculateRhoA1Data(df, len(df_ps))
        else:
            # no need to check here as checked in cleanRecoData
            df_inputs_data, boost = self.calculateA1A1Data(df, len(df_ps))
        # df.to_pickle('misc/debugging_2.pkl')
        # return 
        df_inputs = pd.DataFrame(df_inputs_data)
        if binary:
            df_inputs['y'] = y
        addons_loaded = ""
        if addons:
            self.createAddons(addons, df, df_inputs, binary, addons_config, boost=boost)
            addons_loaded = '_'+'_'.join(addons)
        if save:
            print('Saving df to pickle')
            pickle_file_name = f'{DataLoader.input_df_save_dir}/input_{self.channel}{addons_loaded}'
            if binary:
                pickle_file_name += '_b'
            df_inputs.to_pickle(pickle_file_name+'.pkl')
        return df_inputs

    def calculateRhoRhoData_old(self, df):
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
        pi_1_boosted_rot, pi_2_boosted_rot = [], []
        pi0_1_boosted_rot, pi0_2_boosted_rot = [], []
        rho_1_boosted_rot, rho_2_boosted_rot = [], []
        for i in range(pi_1_boosted[:].shape[1]):
            rot_mat = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0, 0, 1])
            pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
            pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
            pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
            pi0_2_boosted_rot.append(rot_mat.dot(pi0_2_boosted[1:, i]))
            rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
            rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
            if i % 100000 == 0:
                print('finished getting rotated 4-vector', i)
        pi_1_boosted_rot = np.array(pi_1_boosted_rot)
        pi_2_boosted_rot = np.array(pi_2_boosted_rot)
        pi0_1_boosted_rot = np.array(pi0_1_boosted_rot)
        pi0_2_boosted_rot = np.array(pi0_2_boosted_rot)
        rho_1_boosted_rot = np.array(rho_1_boosted_rot)
        rho_2_boosted_rot = np.array(rho_2_boosted_rot)
        df_inputs_data = {
            'pi_E_1_br': pi_1_boosted[0],
            'pi_px_1_br': pi_1_boosted_rot[:, 0],
            'pi_py_1_br': pi_1_boosted_rot[:, 1],
            'pi_pz_1_br': pi_1_boosted_rot[:, 2],
            'pi_E_2_br': pi_2_boosted[0],
            'pi_px_2_br': pi_2_boosted_rot[:, 0],
            'pi_py_2_br': pi_2_boosted_rot[:, 1],
            'pi_pz_2_br': pi_2_boosted_rot[:, 2],
            'pi0_E_1_br': pi0_1_boosted[0],
            'pi0_px_1_br': pi0_1_boosted_rot[:, 0],
            'pi0_py_1_br': pi0_1_boosted_rot[:, 1],
            'pi0_pz_1_br': pi0_1_boosted_rot[:, 2],
            'pi0_E_2_br': pi0_2_boosted[0],
            'pi0_px_2_br': pi0_2_boosted_rot[:, 0],
            'pi0_py_2_br': pi0_2_boosted_rot[:, 1],
            'pi0_pz_2_br': pi0_2_boosted_rot[:, 2],
            'rho_E_1_br': rho_1_boosted[0],
            'rho_px_1_br': rho_1_boosted_rot[:, 0],
            'rho_py_1_br': rho_1_boosted_rot[:, 1],
            'rho_pz_1_br': rho_1_boosted_rot[:, 2],
            'rho_E_2_br': rho_2_boosted[0],
            'rho_px_2_br': rho_2_boosted_rot[:, 0],
            'rho_py_2_br': rho_2_boosted_rot[:, 1],
            'rho_pz_2_br': rho_2_boosted_rot[:, 2],
            'aco_angle_1': df['aco_angle_1'],
            'y_1_1': df['y_1_1'],
            'y_1_2': df['y_1_2'],
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_1': rho_1.m,
            'm_2': rho_2.m,
        }
        return df_inputs_data, boost


    def calculateRhoRhoData(self, df, len_df_ps=0):
        """
        Applies boosting, rotations and calculations to rho_rho channel events
        """
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        # boost into rest frame of resonances
        # rest_frame = pi_1 + pi_2 + pi0_1 + pi0_2
        rest_frame = pi_1 + pi_2
        # boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        boost = -rest_frame
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi0_2_boosted = pi0_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi0_2_boosted
        # rotations
        pi_1_boosted_rot, pi_2_boosted_rot = [], []
        pi0_1_boosted_rot, pi0_2_boosted_rot = [], []
        rho_1_boosted_rot, rho_2_boosted_rot = [], []
        
        want_rotations = True # !!! should be an input parameter
        
        if want_rotations:
            for i in range(pi_1_boosted[:].shape[1]):
                rot_mat = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0, 0, 1])
                pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
                pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
                pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
                pi0_2_boosted_rot.append(rot_mat.dot(pi0_2_boosted[1:, i]))
                rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
                rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
                if i % 100000 == 0:
                    print('finished getting rotated 4-vector', i)
            pi_1_boosted_rot = np.array(pi_1_boosted_rot)
            pi_2_boosted_rot = np.array(pi_2_boosted_rot)
            pi0_1_boosted_rot = np.array(pi0_1_boosted_rot)
            pi0_2_boosted_rot = np.array(pi0_2_boosted_rot)
            rho_1_boosted_rot = np.array(rho_1_boosted_rot)
            rho_2_boosted_rot = np.array(rho_2_boosted_rot)
        else: # if don't want rotations
            pi_1_boosted_rot = pi_1_boosted[1:, :].T
            pi_2_boosted_rot = pi_2_boosted[1:, :].T
            pi0_1_boosted_rot = pi0_1_boosted[1:, :].T
            pi0_2_boosted_rot = pi0_2_boosted[1:, :].T
            rho_1_boosted_rot = rho_1_boosted[1:, :].T
            rho_2_boosted_rot = rho_2_boosted[1:, :].T
        
        def padded(vect3):
            zeros = np.reshape(np.zeros(len(vect3)), (-1, 1))
            return np.concatenate([zeros, vect3], axis=1)
        # print(padded(pi0_1_boosted_rot[:]).shape)
        # print(df['y_1_1'].to_numpy().shape)
        
        # print('df[y_1_1].shape =', df['y_1_1'].shape)
        # print('pi0_1_boosted_rot[:].shape =', pi0_1_boosted_rot[:].shape)
        # print('padded(pi0_1_boosted_rot[:]).shape =', padded(pi0_1_boosted_rot[:]).shape)
        # # aco_angle_1_calc = self.calc_aco_angles(padded(pi0_1_boosted_rot[:]), padded(pi0_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        # print('pi0_1_boosted_rot[:]', pi0_1_boosted_rot[:10, :])
        aco_angle_1_calc = self.calc_aco_angles_alie(padded(pi0_1_boosted_rot[:]), padded(pi0_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        
        # with open('rhorho_aco_angle_calc.txt', 'w') as f:
        #     f.write('\n'.join([str(x) for x in aco_angle_1_calc[:20]]))
        # with open('rhorho_aco_angle_given.txt', 'w') as f:
        #     f.write('\n'.join([str(x) for x in df['aco_angle_1'].to_numpy()[:20]]))
        # print('correct phi stars:', np.sum(np.abs(aco_angle_1_calc - df['aco_angle_1']) < 0.01))
        # print('number of phi stars:', np.sum(np.array(aco_angle_1_calc) < np.inf))
        
        # plt.figure(12)
        # aco_angle_1_calc_ps = aco_angle_1_calc[:len_df_ps]
        # aco_angle_1_calc_sm = aco_angle_1_calc[len_df_ps:]
        # plt.hist(aco_angle_1_calc_ps, bins=50, alpha=0.5)
        # plt.hist(aco_angle_1_calc_sm, bins=50, alpha=0.5)
        
        # plt.figure(15)
        # plt.title('filtered difference between given and calculated aco_angle')
        # df_ps_aco = df['aco_angle_1'][:len_df_ps]
        # df_sm_aco = df['aco_angle_1'][len_df_ps:]
        # diff_ps = aco_angle_1_calc_ps - df_ps_aco.to_numpy()
        # diff_sm = aco_angle_1_calc_sm - df_sm_aco.to_numpy()
        # print('Number of insensible given values:', len([x for x in df_ps_aco if x>9000 or x<-9000]) + len([x for x in df_sm_aco if x>9000 or x<-9000]))
        # print('Mean of insensible given values:', np.mean([x for x in df_ps_aco if x>9000 or x<-9000]))
        # print('Number of calculated nans:', np.sum(np.isnan(aco_angle_1_calc)))
        # print('Incorrect calculations:', np.sum(diff_ps>0.001) + np.sum(diff_sm>0.001))
        # diff_ps = np.array([x for x in diff_ps if x<0.0015 and x>-0.0015])
        # diff_sm = np.array([x for x in diff_sm if x<0.0015 and x>-0.0015])
        # plt.hist(diff_ps, bins=50, alpha=0.5, range=[-1e-12, 1e-12])
        # plt.hist(diff_sm, bins=50, alpha=0.5, range=[-1e-12, 1e-12])
        
        # FOR DEBUGGING:
        # df['pi_E_1_br'] = pi_1_boosted[0]
        # df['pi_px_1_br'] = pi_1_boosted_rot[:, 0]
        # df['pi_py_1_br'] = pi_1_boosted_rot[:, 1]
        # df['pi_pz_1_br'] = pi_1_boosted_rot[:, 2]
        # df['pi_E_2_br'] = pi_2_boosted[0]
        # df['pi_px_2_br'] = pi_2_boosted_rot[:, 0]
        # df['pi_py_2_br'] = pi_2_boosted_rot[:, 1]
        # df['pi_pz_2_br'] = pi_2_boosted_rot[:, 2]
        # df['pi0_E_1_br'] = pi0_1_boosted[0]
        # df['pi0_px_1_br'] = pi0_1_boosted_rot[:, 0]
        # df['pi0_py_1_br'] = pi0_1_boosted_rot[:, 1]
        # df['pi0_pz_1_br'] = pi0_1_boosted_rot[:, 2]
        # df['pi0_E_2_br'] = pi0_2_boosted[0]
        # df['pi0_px_2_br'] = pi0_2_boosted_rot[:, 0]
        # df['pi0_py_2_br'] = pi0_2_boosted_rot[:, 1]
        # df['pi0_pz_2_br'] = pi0_2_boosted_rot[:, 2]
        # df['rho_E_1_br'] = rho_1_boosted[0]
        # df['rho_px_1_br'] = rho_1_boosted_rot[:, 0]
        # df['rho_py_1_br'] = rho_1_boosted_rot[:, 1]
        # df['rho_pz_1_br'] = rho_1_boosted_rot[:, 2]
        # df['rho_E_2_br'] = rho_2_boosted[0]
        # df['rho_px_2_br'] = rho_2_boosted_rot[:, 0]
        # df['rho_py_2_br'] = rho_2_boosted_rot[:, 1]
        # df['rho_pz_2_br'] = rho_2_boosted_rot[:, 2]

        df_inputs_data = {
            'pi_E_1_br': pi_1_boosted[0],
            'pi_px_1_br': pi_1_boosted_rot[:, 0],
            'pi_py_1_br': pi_1_boosted_rot[:, 1],
            'pi_pz_1_br': pi_1_boosted_rot[:, 2],
            'pi_E_2_br': pi_2_boosted[0],
            'pi_px_2_br': pi_2_boosted_rot[:, 0],
            'pi_py_2_br': pi_2_boosted_rot[:, 1],
            'pi_pz_2_br': pi_2_boosted_rot[:, 2],
            'pi0_E_1_br': pi0_1_boosted[0],
            'pi0_px_1_br': pi0_1_boosted_rot[:, 0],
            'pi0_py_1_br': pi0_1_boosted_rot[:, 1],
            'pi0_pz_1_br': pi0_1_boosted_rot[:, 2],
            'pi0_E_2_br': pi0_2_boosted[0],
            'pi0_px_2_br': pi0_2_boosted_rot[:, 0],
            'pi0_py_2_br': pi0_2_boosted_rot[:, 1],
            'pi0_pz_2_br': pi0_2_boosted_rot[:, 2],
            'rho_E_1_br': rho_1_boosted[0],
            'rho_px_1_br': rho_1_boosted_rot[:, 0],
            'rho_py_1_br': rho_1_boosted_rot[:, 1],
            'rho_pz_1_br': rho_1_boosted_rot[:, 2],
            'rho_E_2_br': rho_2_boosted[0],
            'rho_px_2_br': rho_2_boosted_rot[:, 0],
            'rho_py_2_br': rho_2_boosted_rot[:, 1],
            'rho_pz_2_br': rho_2_boosted_rot[:, 2],
            # 'aco_angle_1': df['aco_angle_1'],
            'aco_angle_1': aco_angle_1_calc,
            'y_1_1': df['y_1_1'],
            'y_1_2': df['y_1_2'],
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_1': rho_1.m,
            'm_2': rho_2.m,
        }
        return df_inputs_data, boost
    
    def rotation_matrix(self, axis, theta):
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        axis = np.asarray(axis)
        axis = axis / np.dot(axis, axis)**0.5
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def rotate(self, vect, axis, theta):
        return np.dot(self.rotation_matrix(axis, theta), vect)
    
    def calculate_aco_angles(self, pi_1, pi_2, pi0_1, pi2_2, pi3_2, y1, y2, which_aco_angle='rhoa1-5'):
        p3 = Momentum4(pi_1[:, 0], pi_1[:, 1], pi_1[:, 2], pi_1[:, 3]) # p3 = charged pion 1
        p4 = Momentum4(pi_2[:, 0], pi_2[:, 1], pi_2[:, 2], pi_2[:, 3]) # p4 = charged pion 2
        
        if which_aco_angle == 'rhoa1-5': # this is the same though process as rhoa1-4, but I think it's corrected, because particle 1 and particle 3 are the same composite particle
            pi0 = Momentum4(pi0_1[:, 0], pi0_1[:, 1], pi0_1[:, 2], pi0_1[:, 3]) # pi0 = neutral pion 1
            pi2 = Momentum4(pi2_2[:, 0], pi2_2[:, 1], pi2_2[:, 2], pi2_2[:, 3]) # pi2 = second charged pion 2
            pi3 = Momentum4(pi3_2[:, 0], pi3_2[:, 1], pi3_2[:, 2], pi3_2[:, 3]) # pi3 = third carged pion 3
            # p3 = pi_1
            # p4 = pi_2
            
            # # this gives: good distr for the p4+pi3 combination, but bad distr for the p4+pi2 neutral rho
            # p1 = p3
            # p2 = p4 + pi3
            # p3 = pi0
            # p4 = pi2
            
            p1 = pi0
            p2 = pi2
            p3 = p3
            p4 = p4 + pi3
            
            # # this is the other option:
            # p1 = p3
            # p2 = p4 + pi2
            # p3 = pi0
            # p4 = pi3
        
        return self.calc_aco_angles(p1[:].T, p2[:].T, p3[:].T, p4[:].T, y1, y2)

    def getAcoAngles(self, **kwargs):
        """
        Returns all the aco angles for different channels
        """
        aco_angles = []
        if self.channel == 'rho_rho':
            pi_1_boosted = kwargs['pi_1_boosted']
            pi_2_boosted = kwargs['pi_2_boosted'] 
            pi0_1_boosted = kwargs['pi0_1_boosted']
            pi0_2_boosted = kwargs['pi0_2_boosted']
            aco_angle_1 = self.getAcoAnglesForOneRF(pi0_1_boosted, pi0_2_boosted, pi_1_boosted, pi_2_boosted)
            aco_angles.append(aco_angle_1)
        elif self.channel == 'rho_a1':

            pass
        elif self.channel == 'a1_a1':
            pass
        else:
            raise ValueError('Channel not understood')
        return aco_angles

    def getAcoAnglesForOneRF(self, p1, p2, p3, p4, rest_frame):
        """
        TO TEST: (swapping p1 and p3)
        all inputs are Momentum4
        p1, p3 from same decay
        p2, p4 from same decay
        calculates the angle that spans between p1, p3 plane and p2, p4 plane
        """
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        p1_boosted = p1.boost_particle(boost)
        p2_boosted = p2.boost_particle(boost)
        p3_boosted = p3.boost_particle(boost)
        p4_boosted = p4.boost_particle(boost)
        p1_b_p = np.c_[p1_boosted.p_x, p1_boosted.p_y, p1_boosted.p_z]
        p2_b_p = np.c_[p2_boosted.p_x, p2_boosted.p_y, p2_boosted.p_z]
        p3_b_p = np.c_[p3_boosted.p_x, p3_boosted.p_y, p3_boosted.p_z]
        p4_b_p = np.c_[p4_boosted.p_x, p4_boosted.p_y, p4_boosted.p_z]
        n1 = p1_b_p - np.multiply(np.einsum('ij, ij->i', p1_b_p, self.normaliseVector(p3_b_p))[:, None], self.normaliseVector(p3_b_p))
        n2 = p2_b_p - np.multiply(np.einsum('ij, ij->i', p2_b_p, self.normaliseVector(p4_b_p))[:, None], self.normaliseVector(p4_b_p))
        # vectorised form of
        # n1 = p1.Vect() - p1.Vect().Dot(p3.Vect().Unit())*p3.Vect().Unit();    
        # n2 = p2.Vect() - p2.Vect().Dot(p4.Vect().Unit())*p4.Vect().Unit();
        return np.arccos(np.einsum('ij, ij->i', n1, n2))

    def normaliseVector(vec):
        return np.sqrt(np.einsum('...i,...i', vec, vec))

    def calc_aco_angles(self, pp1, pp2, pp3, pp4, yy1, yy2):
        angles = []
        for i in range(len(pp1)):
            p3 = pp3[i]
            p4 = pp4[i]
            p1 = pp1[i]
            p2 = pp2[i]
            y1 = yy1[i]
            y2 = yy2[i]
            # print(p3.shape)
    
            def unit(vect):
                return vect / np.linalg.norm(vect)
            
            n1 = p1[1:] - np.dot(p1[1:], unit(p3[1:])) * unit(p3[1:])
            n2 = p2[1:] - np.dot(p2[1:], unit(p4[1:])) * unit(p4[1:])
            n1 = unit(n1)
            n2 = unit(n2)
    
            angle = np.arccos(np.dot(n1, n2))
            # print(p4.shape)
            # print(n1.shape)
            # print(n2.shape)
            sign = np.dot(unit(p4[1:]), np.cross(n1, n2))
    
            # shift 1
            if sign < 0:
                angle = 2 * np.pi - angle
    
            # shift 2
            if y1*y2 < 0:
                if angle < np.pi:
                    angle += np.pi
                else:
                    angle -= np.pi
    
            angles.append(angle)
    
            if i%100000==0:
                print('finished element', i)
                
        return angles
    
    def calc_aco_angles_alie(self, pp1, pp2, pp3, pp4, yy1, yy2):
        pp1 = pp1.T
        pp2 = pp2.T
        pp3 = pp3.T
        pp4 = pp4.T
        
        print('shape of pp1:', pp1.shape)
        print('shape of yy1:', yy1.shape)
        
        #Some geometrical functions
        def cross_product(vector3_1,vector3_2):
            if len(vector3_1)!=3 or len(vector3_1)!=3:
                print('These are not 3D arrays !')
            x_perp_vector=vector3_1[1]*vector3_2[2]-vector3_1[2]*vector3_2[1]
            y_perp_vector=vector3_1[2]*vector3_2[0]-vector3_1[0]*vector3_2[2]
            z_perp_vector=vector3_1[0]*vector3_2[1]-vector3_1[1]*vector3_2[0]
            return np.array([x_perp_vector,y_perp_vector,z_perp_vector])
        
        def dot_product(vector1,vector2):
            if len(vector1)!=len(vector2):
                print('vector1 =', vector1)
                print('vector2 =', vector2)
                raise Exception('Arrays_of_different_size')
            prod=0
            for i in range(len(vector1)):
                prod=prod+vector1[i]*vector2[i]
            return prod
    
        def norm(vector):
            if len(vector)!=3:
                print('This is only for a 3d vector')
            return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)
        
        #calculating the perpependicular component
        pi0_1_3Mom_star_perp=cross_product(pp1[1:], pp3[1:])
        pi0_2_3Mom_star_perp=cross_product(pp2[1:], pp4[1:])
        
        #Now normalise:
        pi0_1_3Mom_star_perp=pi0_1_3Mom_star_perp/norm(pi0_1_3Mom_star_perp)
        pi0_2_3Mom_star_perp=pi0_2_3Mom_star_perp/norm(pi0_2_3Mom_star_perp)
        
        #Calculating phi_star
        phi_CP=np.arccos(dot_product(pi0_1_3Mom_star_perp,pi0_2_3Mom_star_perp))
        
        #The energy ratios
        y_T = np.array(yy1 * yy2)
        
        #Up to here I agree with Kingsley
        print('phi_CP[:10]', phi_CP[:10],'\n')
        
        #The O variable
        cross=np.cross(pi0_1_3Mom_star_perp.transpose(),pi0_2_3Mom_star_perp.transpose()).transpose()
        bigO=dot_product(pp4[1:],cross)
        
        #perform the shift w.r.t. O* sign
        phi_CP=np.where(bigO>=0, 2*np.pi-phi_CP, phi_CP)#, phi_CP)
        
        #additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
        #phi_CP=np.where(y_T<0, 2*np.pi-phi_CP, np.pi-phi_CP)
        phi_CP=np.where(y_T>=0, np.where(phi_CP<np.pi, phi_CP+np.pi, phi_CP-np.pi), phi_CP)
    
        return phi_CP    

    def calculateRhoA1Data(self, df, len_df_ps=0):
        # TODO: kristof implement:
        # - under construction!
        # - need to add other aco_angles calculation code
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
        pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
        rho_1 = pi_1 + pi0_1 # charged rho
        rho_2 = pi_2 + pi3_2 # neutral rho, a part of the charged a1 particle
        a1 = rho_2 + pi2_2
        # boost into rest frame of resonances
        # rest_frame = pi_1 + pi_2 + pi0_1 + pi2_2 + pi3_2
        # rest_frame = pi0_1 + pi_1 + pi_2
        rest_frame = pi_1 + pi_2
        # boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        boost = - rest_frame
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi2_2_boosted = pi2_2.boost_particle(boost)
        pi3_2_boosted = pi3_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi3_2_boosted
        a1_boosted = rho_2_boosted + pi2_2_boosted
        rest_frame_boosted = pi_1_boosted + pi_2_boosted + pi0_1_boosted
        # rest_frame_boosted = rest_frame.boost_particle(boost)
        
        want_rotations = True # !!! Maybe this should be an input parameter
        
        # rotations
        if want_rotations:
            pi_1_boosted_rot, pi_2_boosted_rot = [], []
            pi0_1_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot = [], [], []
            rho_1_boosted_rot, rho_2_boosted_rot, a1_boosted_rot = [], [], []
            
            # MY ROTATIONS:
            # unit vectors along the momenta of the primary resonances
            unit1 = (rho_1_boosted[1:, :] / np.linalg.norm(rho_1_boosted[1:, :], axis=0)).transpose()
            unit2 = (pi_2_boosted[1:, :] / np.linalg.norm(pi_2_boosted[1:, :], axis=0)).transpose()
            # probably there's a faster way of doing this
            zaxis = np.array([np.array([0., 0., 1.]) for _ in range(len(unit1))])
            axes1 = np.cross(unit1, zaxis)
            axes2 = np.cross(unit2, zaxis)
            dotproduct1 = (unit1*zaxis).sum(1)
            angles1 = np.arccos(dotproduct1)
            dotproduct2 = (unit2*zaxis).sum(1)
            angles2 = np.arccos(dotproduct2)
            
            for i in range(pi_1_boosted[:].shape[1]):
# =============================================================================
#                 # STANLEY'S ROTATIONS:
#                 # rot_mat = self.rotation_matrix_from_vectors(rho_1_boosted[1:, i], [0, 0, 1])
#                 # rot_mat = self.rotation_matrix_from_vectors(pi_1_boosted[1:, i]+pi_2_boosted[1:, i], [0, 0, 1])
#                 # rot_mat = self.rotation_matrix_from_vectors(a1_boosted[1:, i], [0, 0, 1])
#                 rot_mat = self.rotation_matrix_from_vectors(rest_frame_boosted[1:, i], [0, 0, 1])
#                 pi_1_boosted_rot.append(rot_mat.dot(pi_1_boosted[1:, i]))
#                 pi0_1_boosted_rot.append(rot_mat.dot(pi0_1_boosted[1:, i]))
#                 pi_2_boosted_rot.append(rot_mat.dot(pi_2_boosted[1:, i]))
#                 pi2_2_boosted_rot.append(rot_mat.dot(pi2_2_boosted[1:, i]))
#                 pi3_2_boosted_rot.append(rot_mat.dot(pi3_2_boosted[1:, i]))
#                 rho_1_boosted_rot.append(rot_mat.dot(rho_1_boosted[1:, i]))
#                 rho_2_boosted_rot.append(rot_mat.dot(rho_2_boosted[1:, i]))
#                 a1_boosted_rot.append(rot_mat.dot(a1_boosted[1:, i]))
# =============================================================================
                
                # MY ROTATIONS:
                pi_1_boosted_rot.append(self.rotate(pi_1_boosted[1:, i], axes1[i], angles1[i]))
                pi0_1_boosted_rot.append(self.rotate(pi0_1_boosted[1:, i], axes1[i], angles1[i]))
                pi_2_boosted_rot.append(self.rotate(pi_2_boosted[1:, i], axes1[i], angles1[i]))
                pi2_2_boosted_rot.append(self.rotate(pi2_2_boosted[1:, i], axes1[i], angles1[i]))
                pi3_2_boosted_rot.append(self.rotate(pi3_2_boosted[1:, i], axes1[i], angles1[i]))
                rho_1_boosted_rot.append(self.rotate(rho_1_boosted[1:, i], axes1[i], angles1[i]))
                rho_2_boosted_rot.append(self.rotate(rho_2_boosted[1:, i], axes1[i], angles1[i]))
                a1_boosted_rot.append(self.rotate(a1_boosted[1:, i], axes1[i], angles1[i]))
                
                if i % 100000 == 0:
                    print('finished getting rotated 4-vector', i)
            pi_1_boosted_rot = np.array(pi_1_boosted_rot)
            pi_2_boosted_rot = np.array(pi_2_boosted_rot)
            pi0_1_boosted_rot = np.array(pi0_1_boosted_rot)
            pi2_2_boosted_rot = np.array(pi2_2_boosted_rot)
            pi3_2_boosted_rot = np.array(pi3_2_boosted_rot)
            rho_1_boosted_rot = np.array(rho_1_boosted_rot)
            rho_2_boosted_rot = np.array(rho_2_boosted_rot)
            a1_boosted_rot = np.array(a1_boosted_rot)
            
            # write out some rotated 4-vectors to a file, to compare with shared code
            print('started writing out 4-vectors')
            with open('4vectors/rotated_4vectors.txt', 'w') as f:
                for i in range(10):
                    p1str = ' '.join([str(x) for x in pi0_1_boosted_rot[i]])
                    p3str = ' '.join([str(x) for x in pi_1_boosted_rot[i]])
                    p4str = ' '.join([str(x) for x in pi_2_boosted_rot[i]])
                    f.write(p1str+'\t\t'+p3str+'\t\t'+p4str+'\n')
            print('finished writing out 4-vectors')
            
            def padded(vect3):
                zeros = np.reshape(np.zeros(len(vect3)), (-1, 1))
                return np.concatenate([zeros, vect3], axis=1)
            # zeros_1 = np.reshape(np.zeros(len(pi_1_boosted_rot)), (-1, 1))
            # zeros_2 = np.reshape(np.zeros(len(pi_2_boosted_rot)), (-1, 1))
            # zeros_3 = np.reshape(np.zeros(len(pi0_1_boosted_rot)), (-1, 1))
            # zeros_4 = np.reshape(np.zeros(len(pi2_2_boosted_rot)), (-1, 1))
            # zeros_5 = np.reshape(np.zeros(len(pi3_2_boosted_rot)), (-1, 1))
            # aco_angle_2 = self.calculate_aco_angles(np.concatenate([zeros_1, pi_1_boosted_rot], axis=1), np.concatenate([zeros_1, pi_2_boosted_rot], axis=1), np.concatenate([zeros_1, pi0_1_boosted_rot], axis=1), np.concatenate([zeros_1, pi2_2_boosted_rot], axis=1), np.concatenate([zeros_1, pi3_2_boosted_rot], axis=1), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
            
            # WARNING! calculate_aco_angles takes the input arguments in a different order than calc_aco_angles
            # aco_angle_2 = self.calculate_aco_angles(padded(pi_1_boosted_rot), padded(pi_2_boosted_rot), padded(pi0_1_boosted_rot), padded(pi2_2_boosted_rot), padded(pi3_2_boosted_rot), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
            # the next line is the interesting one:
            # aco_angle_2 = self.calc_aco_angles_alie(padded(pi0_1_boosted_rot), padded(pi3_2_boosted_rot), padded(pi_1_boosted_rot), padded(pi_2_boosted_rot + pi2_2_boosted_rot), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
            # aco_angle_danny = self.calc_aco_angles(padded(pi0_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi2_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
            aco_angle_danny = self.calc_aco_angles_alie(padded(pi0_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), padded(pi_1_boosted_rot[:]), padded(pi2_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
            aco_angle_2 = aco_angle_danny
            # aco_angle_danny = np.array(aco_angle_danny)
            aco_angle_2 = np.array(aco_angle_2)
            # aco_angle_danny[np.isnan(aco_angle_danny)] = np.pi
            aco_angle_2[np.isnan(aco_angle_2)] = np.pi
            
            # plt.figure(12)
            # aco_angle_2_ps = aco_angle_2[:len_df_ps]
            # aco_angle_2_sm = aco_angle_2[len_df_ps:]
            # plt.hist(aco_angle_2_ps, bins=50, alpha=0.5)
            # plt.hist(aco_angle_2_sm, bins=50, alpha=0.5)
            
            # plt.figure(15)
            # plt.title('filtered difference between given and calculated aco_angle')
            # df_ps_aco = df['aco_angle_1'][:len_df_ps]
            # df_sm_aco = df['aco_angle_1'][len_df_ps:]
            # diff_ps = aco_angle_2_ps - df_ps_aco.to_numpy()
            # diff_sm = aco_angle_2_sm - df_sm_aco.to_numpy()
            # print('Number of insensible given values:', len([x for x in df_ps_aco if x>9000 or x<-9000]) + len([x for x in df_sm_aco if x>9000 or x<-9000]))
            # print('Mean of insensible given values:', np.mean([x for x in df_ps_aco if x>9000 or x<-9000]))
            # print('Number of calculated nans:', np.sum(np.isnan(aco_angle_danny)))
            # print('Incorrect calculations:', np.sum(diff_ps>0.001) + np.sum(diff_sm>0.001))
            # diff_ps = np.array([x for x in diff_ps if x<0.0015 and x>-0.0015])
            # diff_sm = np.array([x for x in diff_sm if x<0.0015 and x>-0.0015])
            # plt.hist(diff_ps, bins=50, alpha=0.5)
            # plt.hist(diff_sm, bins=50, alpha=0.5)
            
        else: # if don't want rotations:
            pi_1_boosted_rot = np.array(pi_1_boosted).T
            pi_2_boosted_rot = np.array(pi_2_boosted).T
            pi0_1_boosted_rot = np.array(pi0_1_boosted).T
            pi2_2_boosted_rot = np.array(pi2_2_boosted).T
            pi3_2_boosted_rot = np.array(pi3_2_boosted).T
            rho_1_boosted_rot = np.array(rho_1_boosted).T
            rho_2_boosted_rot = np.array(rho_2_boosted).T
            a1_boosted_rot = np.array(a1_boosted).T
            
        # print('correct phi stars:', np.sum(np.abs(aco_angle_2 - df['aco_angle_1']) < 0.01))
        # print('number of phi stars:', np.sum(np.array(aco_angle_2) < np.inf))
        # print('number of nans:', np.sum(np.isnan(aco_angle_2)))
        pi_1_boosted_rot[np.isnan(pi_1_boosted_rot)] == np.mean(pi_1_boosted_rot)
        pi_2_boosted_rot[np.isnan(pi_2_boosted_rot)] == np.mean(pi_2_boosted_rot)
        pi0_1_boosted_rot[np.isnan(pi0_1_boosted_rot)] == np.mean(pi0_1_boosted_rot)
        pi2_2_boosted_rot[np.isnan(pi2_2_boosted_rot)] == np.mean(pi2_2_boosted_rot)
        pi3_2_boosted_rot[np.isnan(pi3_2_boosted_rot)] == np.mean(pi3_2_boosted_rot)
        
        df_inputs_data = {
            'pi_E_1_br': pi_1_boosted[0],
            'pi_px_1_br': pi_1_boosted_rot[:, 0],
            'pi_py_1_br': pi_1_boosted_rot[:, 1],
            'pi_pz_1_br': pi_1_boosted_rot[:, 2],
            'pi_E_2_br': pi_2_boosted[0],
            'pi_px_2_br': pi_2_boosted_rot[:, 0],
            'pi_py_2_br': pi_2_boosted_rot[:, 1],
            'pi_pz_2_br': pi_2_boosted_rot[:, 2],
            'pi0_E_1_br': pi0_1_boosted[0],
            'pi0_px_1_br': pi0_1_boosted_rot[:, 0],
            'pi0_py_1_br': pi0_1_boosted_rot[:, 1],
            'pi0_pz_1_br': pi0_1_boosted_rot[:, 2],
            'pi2_E_2_br': pi2_2_boosted[0],
            'pi2_px_2_br': pi2_2_boosted_rot[:, 0],
            'pi2_py_2_br': pi2_2_boosted_rot[:, 1],
            'pi2_pz_2_br': pi2_2_boosted_rot[:, 2],
            'pi3_E_2_br': pi3_2_boosted[0],
            'pi3_px_2_br': pi3_2_boosted_rot[:, 0],
            'pi3_py_2_br': pi3_2_boosted_rot[:, 1],
            'pi3_pz_2_br': pi3_2_boosted_rot[:, 2],
            'rho_E_1_br': rho_1_boosted[0],
            'rho_px_1_br': rho_1_boosted_rot[:, 0],
            'rho_py_1_br': rho_1_boosted_rot[:, 1],
            'rho_pz_1_br': rho_1_boosted_rot[:, 2],
            'rho_E_2_br': rho_2_boosted[0],
            'rho_px_2_br': rho_2_boosted_rot[:, 0],
            'rho_py_2_br': rho_2_boosted_rot[:, 1],
            'rho_pz_2_br': rho_2_boosted_rot[:, 2],
            'a1_E_br': a1_boosted[0],
            'a1_px_br': a1_boosted_rot[:, 0],
            'a1_py_br': a1_boosted_rot[:, 1],
            'a1_pz_br': a1_boosted_rot[:, 2],
            # 'aco_angle_1': df['aco_angle_1'],
            # 'aco_angle_1': aco_angle_danny,
            'aco_angle_1': aco_angle_2,
            # 'aco_angle_2': aco_angle_2,
            'y_1_1': df['y_1_1'],
            'y_1_2': df['y_1_2'],
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_1': rho_1.m,
            #'m_2': rho_2.m,
            'm_2': a1.m,
        }
        return df_inputs_data, boost

    def calculateA1A1Data(self, df, len_df_ps=0):
        # TODO: include this channel
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi2_1 = Momentum4(df['pi2_E_1'], df["pi2_px_1"], df["pi2_py_1"], df["pi2_pz_1"])
        pi3_1 = Momentum4(df['pi3_E_1'], df["pi3_px_1"], df["pi3_py_1"], df["pi3_pz_1"])
        pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
        pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
        
        rest_frame = pi_1 + pi_2
        # boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        boost = - rest_frame
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi2_1_boosted = pi2_1.boost_particle(boost)
        pi3_1_boosted = pi3_1.boost_particle(boost)
        pi2_2_boosted = pi2_2.boost_particle(boost)
        pi3_2_boosted = pi3_2.boost_particle(boost)
        
        rho_1 = pi_1_boosted + pi2_1_boosted
        a1 = rho_1 + pi3_1_boosted
        rho_2 = pi_2_boosted + pi2_2_boosted
        a2 = rho_2 + pi3_2_boosted
        
        # rotations
        # not doing properly rotations, just reassign variables
        pi_1_boosted_rot = pi_1_boosted[1:, :].T
        pi_2_boosted_rot = pi_2_boosted[1:, :].T
        pi2_1_boosted_rot = pi2_1_boosted[1:, :].T
        pi2_2_boosted_rot = pi2_2_boosted[1:, :].T
        pi3_1_boosted_rot = pi3_1_boosted[1:, :].T
        pi3_2_boosted_rot = pi3_2_boosted[1:, :].T

        def padded(vect3):
            zeros = np.reshape(np.zeros(len(vect3)), (-1, 1))
            return np.concatenate([zeros, vect3], axis=1)        
        aco_angle_2 = self.calc_aco_angles_alie(padded(pi_1_boosted_rot[:]), padded(pi_2_boosted_rot[:]), padded(pi2_1_boosted_rot[:]), padded(pi2_2_boosted_rot[:]), df['y_1_1'].to_numpy(), df['y_1_2'].to_numpy())
        aco_angle_2 = np.array(aco_angle_2)
        aco_angle_2[np.isnan(aco_angle_2)] = np.pi
            
        plt.figure(12)
        aco_angle_2_ps = aco_angle_2[:len_df_ps]
        aco_angle_2_sm = aco_angle_2[len_df_ps:]
        plt.hist(aco_angle_2_ps, bins=50, alpha=0.5)
        plt.hist(aco_angle_2_sm, bins=50, alpha=0.5)
        
        plt.figure(15)
        plt.title('filtered difference between given and calculated aco_angle')
        df_ps_aco = df['aco_angle_1'][:len_df_ps]
        df_sm_aco = df['aco_angle_1'][len_df_ps:]
        diff_ps = aco_angle_2_ps - df_ps_aco.to_numpy()
        diff_sm = aco_angle_2_sm - df_sm_aco.to_numpy()
        print('Number of insensible given values:', len([x for x in df_ps_aco if x>9000 or x<-9000]) + len([x for x in df_sm_aco if x>9000 or x<-9000]))
        print('Mean of insensible given values:', np.mean([x for x in df_ps_aco if x>9000 or x<-9000]))
        print('Number of calculated nans:', np.sum(np.isnan(aco_angle_2)))
        print('Incorrect calculations:', np.sum(diff_ps>0.001) + np.sum(diff_sm>0.001))
        diff_ps = np.array([x for x in diff_ps if x<0.0015 and x>-0.0015])
        diff_sm = np.array([x for x in diff_sm if x<0.0015 and x>-0.0015])
        plt.hist(diff_ps, bins=50, alpha=0.5)
        plt.hist(diff_sm, bins=50, alpha=0.5)
        
        df_inputs_data = {
            'pi_E_1_br': pi_1_boosted[0],
            'pi_px_1_br': pi_1_boosted_rot[:, 0],
            'pi_py_1_br': pi_1_boosted_rot[:, 1],
            'pi_pz_1_br': pi_1_boosted_rot[:, 2],
            'pi_E_2_br': pi_2_boosted[0],
            'pi_px_2_br': pi_2_boosted_rot[:, 0],
            'pi_py_2_br': pi_2_boosted_rot[:, 1],
            'pi_pz_2_br': pi_2_boosted_rot[:, 2],
            
            'pi2_E_1_br': pi2_1_boosted[0],
            'pi2_px_1_br': pi2_1_boosted_rot[:, 0],
            'pi2_py_1_br': pi2_1_boosted_rot[:, 1],
            'pi2_pz_1_br': pi2_1_boosted_rot[:, 2],
            'pi3_E_1_br': pi3_1_boosted[0],
            'pi3_px_1_br': pi3_1_boosted_rot[:, 0],
            'pi3_py_1_br': pi3_1_boosted_rot[:, 1],
            'pi3_pz_1_br': pi3_1_boosted_rot[:, 2],
            
            'pi2_E_2_br': pi2_2_boosted[0],
            'pi2_px_2_br': pi2_2_boosted_rot[:, 0],
            'pi2_py_2_br': pi2_2_boosted_rot[:, 1],
            'pi2_pz_2_br': pi2_2_boosted_rot[:, 2],
            'pi3_E_2_br': pi3_2_boosted[0],
            'pi3_px_2_br': pi3_2_boosted_rot[:, 0],
            'pi3_py_2_br': pi3_2_boosted_rot[:, 1],
            'pi3_pz_2_br': pi3_2_boosted_rot[:, 2],
            # 'aco_angle_1': df['aco_angle_1'],
            # 'aco_angle_1': aco_angle_danny,
            'aco_angle_1': aco_angle_2,
            # 'aco_angle_2': aco_angle_2,
            'y_1_1': df['y_1_1'],
            'y_1_2': df['y_1_2'],
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
             'm_1': a1.m,
             'm_2': a2.m,
        }
        return df_inputs_data, boost

    def createAddons(self, addons, df, df_inputs, binary, addons_config={}, **kwargs):
        """
        If you want to create more addon features, put the necessary arguments through kwargs, 
        unpack them at the start of this function, and add an if case to your needs
        TODO: need to catch incorrectly loaded kwargs
        """
        boost = None
        if kwargs:
            boost = kwargs["boost"]
        for addon in addons:
            if addon == 'met' and boost is not None:
                print('Addon MET loaded')
                E_miss, E_miss_x, E_miss_y = self.addonMET(df, boost)
                df_inputs['E_miss'] = E_miss
                df_inputs['E_miss_x'] = E_miss_x
                df_inputs['E_miss_y'] = E_miss_y
            if addon == 'neutrino':
                print('Addon neutrino loaded')
                load_alpha = addons_config['neutrino']['load_alpha']
                termination = addons_config['neutrino']['termination']
                alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = self.addonNeutrinos(df, df_inputs, binary, load_alpha, termination=termination)
                df_inputs['alpha_1'] = alpha_1
                df_inputs['alpha_2'] = alpha_2
                df_inputs['E_nu_1'] = E_nu_1
                df_inputs['E_nu_2'] = E_nu_2
                df_inputs['p_t_nu_1'] = p_t_nu_1
                df_inputs['p_t_nu_2'] = p_t_nu_2
                df_inputs['p_z_nu_1'] = p_z_nu_1
                df_inputs['p_z_nu_2'] = p_z_nu_2

    def addonMET(self, df, boost):
        """
        Addon configuration for the MET
        """
        N = len(df['metx'])
        met_x = Momentum4(df['metx'], np.zeros(N), np.zeros(N), np.zeros(N))
        met_y = Momentum4(df['mety'], np.zeros(N), np.zeros(N), np.zeros(N))
        met = Momentum4(df['met'], np.zeros(N), np.zeros(N), np.zeros(N))
        # boost MET - E_miss is already boosted into the hadronic rest frame
        E_miss = met_x.boost_particle(boost)[0]
        E_miss_x = met_y.boost_particle(boost)[0]
        E_miss_y = met.boost_particle(boost)[0]
        return E_miss, E_miss_x, E_miss_y

    def addonNeutrinos(self, df, df_inputs, binary, load_alpha, termination=100):
        """
        Addon configuration for neutrino information
        TODO:
        - load in neutrino phis
        Returns: alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2
        """
        NR = NeutrinoReconstructor(binary=binary)
        return NR.runAlphaReconstructor(df.reset_index(drop=False), df_inputs.reset_index(drop=False), load_alpha=load_alpha, termination=termination)

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

if __name__ == '__main__':
    variables_rho_rho = [
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
    DL = DataLoader(variables_rho_rho, 'rho_rho')
    # DL.createRecoData(binary=True, addons=['met'])
    # DL.readGenData()
