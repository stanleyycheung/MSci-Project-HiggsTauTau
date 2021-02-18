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
    - 'ip'
    - 'sv'

    """

    reco_root_path = "./MVAFILE_AllHiggs_tt.root"
    gen_root_path = "./MVAFILE_GEN_AllHiggs_tt.root"
    reco_df_path = './df_tt'
    gen_df_path = './df_tt_gen'
    if os.path.exists("C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"):
        print('Using Kristof\'s .root file')
        reco_root_path = "C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"
    if os.path.exists('/home/hep/shc3117/'):
        print('Running on Imperial HEP LX machines')
        reco_root_path = "/vols/cms/shc3117/MVAFILE_AllHiggs_tt.root"
        gen_root_path = "/vols/cms/shc3117/MVAFILE_GEN_AllHiggs_tt.root"
        reco_df_path = '/vols/cms/shc3117/df_tt'
        gen_df_path = '/vols/cms/shc3117/df_tt_gen'
    input_df_save_dir_reco = './input_df_reco'
    input_df_save_dir_gen = './input_df_gen'

    def __init__(self, variables, channel, gen):
        """
        DataLoader should be near stateless, exceptions of the channel and variables needed to load
        Other instance variables should only deal with load/save directories
        """
        self.channel = channel
        self.variables = variables
        self.gen = gen

    def loadRecoData(self, binary, addons=[]):
        """
        Loads the BR df directly from HDF5 - no need to read from .root, boost and rotate events
        """
        print('Reading reco df HDF5 file')
        addons_loaded = ""
        if addons:
            addons_loaded = '_'+'_'.join(addons)
        hdf_file_name = f'{DataLoader.input_df_save_dir_reco}/input_{self.channel}{addons_loaded}'
        if binary:
            hdf_file_name += '_b'
        df_inputs = pd.read_hdf(hdf_file_name+'.h5', 'df')
        return df_inputs

    def loadGenData(self, binary, addons=[]):
        """
        Loads the BR df (gen) directly from HDF5 - no need to read from .root, boost and rotate events
        """
        print('Reading gen df HDF5 file')
        addons_loaded = ""
        if addons:
            addons_loaded = '_'+'_'.join(addons)
        hdf_file_name = f'{DataLoader.input_df_save_dir_gen}/input_gen_{self.channel}{addons_loaded}'
        if binary:
            hdf_file_name += '_b'
        df_inputs = pd.read_hdf(hdf_file_name+'.h5', 'df')
        return df_inputs

    def createRecoData(self, binary, from_hdf=True, addons=[], addons_config={}):
        """
        Creates the input (reco) data for the NN either from .root file or a previously saved .h5 file
        """
        print(f'Loading .root info with using HDF5 as {from_hdf}')
        df = self.readRecoData(from_hdf=from_hdf)
        print('Cleaning data')
        df_clean, df_ps_clean, df_sm_clean = self.cleanRecoData(df)
        print('Creating input data')
        df_inputs = self.createTrainTestData(df_clean, df_ps_clean, df_sm_clean, binary, False, addons, addons_config, save=True)
        return df_inputs

    def createGenData(self, binary, from_hdf=False, addons=[], addons_config={}):
        print(f'Loading .root info with using HDF5 as {from_hdf}')
        df = self.readGenData(from_hdf=from_hdf)
        print('Cleaning data')
        df_clean, df_ps_clean, df_sm_clean = self.cleanGenData(df)
        print('Creating input data')
        df_inputs = self.createTrainTestData(df_clean, df_ps_clean, df_sm_clean, binary, True, addons, addons_config, save=True)
        return df_inputs

    def readRecoData(self, from_hdf=False):
        """
        Reads the reco root file, can save contents into .h5 for fast read/write abilities
        """
        if not from_hdf:
            tree_tt = uproot.open(DataLoader.reco_root_path)["ntuple"]
            # tree_et = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_et.root")["ntuple"]
            # tree_mt = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_mt.root")["ntuple"]
            df = tree_tt.pandas.df(self.variables)
            df.to_hdf(f"{DataLoader.reco_df_path}_{self.channel}.h5", 'df')
        else:
            df = pd.read_hdf(f"{DataLoader.reco_df_path}_{self.channel}.h5", 'df')
        return df

    def readGenData(self, from_hdf=False):
        """
        Reads the gen root file, can save contents into .h5 for fast read/write abilities
        """
        if not from_hdf:
            tree_tt = uproot.open(DataLoader.gen_root_path)["ntuple"]
            df = tree_tt.pandas.df(self.variables)
            df.to_hdf(f"{DataLoader.gen_df_path}_{self.channel}.h5", 'df')
        else:
            df = pd.read_hdf(f"{DataLoader.gen_df_path}_{self.channel}.h5", 'df')
        return df

    def cleanGenData(self, df):
        """
        Selects correct channel for gen data
        """
        if self.channel == 'rho_rho':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 1)]
        elif self.channel == 'rho_a1':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 10)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['dm_1'] == 10) & (df['dm_2'] == 10)]
        else:
            raise ValueError('Incorrect channel inputted')
        df_clean = df_clean.dropna()
        df_clean = df_clean.loc[~(df_clean == 0).all(axis=1)]
        df_rho_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
        df_rho_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
        return df_clean, df_rho_ps, df_rho_sm

    def cleanRecoData(self, df):
        """
        Selects correct channel for reco data, whilst seperating sm/ps distributions as well
        TODO: remove 0s in events
        """
        if self.channel == 'rho_rho':
            # select only rho-rho events
            df_clean = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 1) & (df["tau_decay_mode_1"] == 1) & (df["tau_decay_mode_2"] == 1)]
            # drop unnecessary labels
            # df_clean = df_rho.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        elif self.channel == 'rho_a1':
            df_clean = df[(df['mva_dm_1'] == 1) & (df['mva_dm_2'] == 10) & (df["tau_decay_mode_1"] == 1)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['mva_dm_1'] == 10) & (df['mva_dm_2'] == 10)]
            # removing events with 0s in them
            # df_clean = df_clean.loc[~(df_clean['pi_px_1'] == 0)]
        else:
            raise ValueError('Incorrect channel inputted')
        # removing all 0s in df
        # df.loc[(df!=0).any(1)]
        # select ps and sm data
        df_clean = df_clean.dropna()
        df_clean = df_clean[(df_clean != 0).all(1)]
        df_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
        df_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
        return df_clean, df_ps, df_sm

    def augmentDfToBinary(self, df_ps, df_sm):
        y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
        y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))
        y = pd.concat([y_sm, y_ps]).to_numpy()
        df = pd.concat([df_sm, df_ps]).reset_index(drop=True)
        return df, y

    def createTrainTestData(self, df, df_ps, df_sm, binary, gen, addons, addons_config, save=True):
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
            df_inputs_data = self.calculateRhoRhoData(df)
        elif self.channel == 'rho_a1':
            df_inputs_data = self.calculateRhoA1Data(df)
        else:
            # no need to check here as checked in cleanRecoData
            df_inputs_data = self.calculateA1A1Data(df)
        df_inputs = pd.DataFrame(df_inputs_data)
        if binary:
            df_inputs['y'] = y
        addons_loaded = ""
        if addons:
            df_inputs = self.createAddons(addons, df, df_inputs, binary, addons_config)
            addons_loaded = '_'+'_'.join(addons)
        if save:
            if not gen:
                hdf_file_name = f'{DataLoader.input_df_save_dir_reco}/input_{self.channel}{addons_loaded}'
            else:
                hdf_file_name = f'{DataLoader.input_df_save_dir_gen}/input_gen_{self.channel}{addons_loaded}'
            if binary:
                hdf_file_name += '_b'
            # hdf_file_name = './alpha_analysis/df_br'
            print(f'Saving df to {hdf_file_name}')
            df_inputs.to_hdf(hdf_file_name+'.h5', key='df')
        return df_inputs

    def calculateRhoRhoData(self, df):
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        # boost into rest frame of resonances
        rest_frame = rho_1 + rho_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi0_2_boosted = pi0_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi0_2_boosted
        br_vectors = self.rotateVectors(pi_1=pi_1_boosted, pi_2=pi_2_boosted, pi0_1=pi0_1_boosted, pi0_2=pi0_2_boosted)
        pi_1_boosted_rot, pi0_1_boosted_rot, pi_2_boosted_rot, pi0_2_boosted_rot = br_vectors
        rho_1_boosted_rot, rho_2_boosted_rot = pi_1_boosted_rot+pi0_1_boosted_rot, pi_2_boosted_rot+pi0_2_boosted_rot
        # aco angle calculation
        aco_angle_1 = self.getAcoAngles(pi_1=pi_1, pi0_1=pi0_1, pi_2=pi_2, pi0_2=pi0_2)
        y_rho_1, y_rho_2 = self.getY(pi_1=pi_1, pi0_1=pi0_1, pi_2=pi_2, pi0_2=pi0_2)
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
            'aco_angle_1_calc': aco_angle_1,
            'y_rho_1': y_rho_1,
            'y_rho_2': y_rho_2,
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_rho_1': rho_1.m,
            'm_rho_2': rho_2.m,
            # for alpha error graphs
            # "gen_nu_p_1": df.gen_nu_p_1,
            # "gen_nu_phi_1": df.gen_nu_phi_1,
            # "gen_nu_eta_1": df.gen_nu_eta_1,
            # "gen_nu_p_2": df.gen_nu_p_2,
            # "gen_nu_phi_2": df.gen_nu_phi_2,
            # "gen_nu_eta_2": df.gen_nu_eta_2,
            # 'pi_E_1': pi_1[0],
            # 'pi_px_1': pi_1[1],
            # 'pi_py_1': pi_1[2],
            # 'pi_pz_1': pi_1[3],
            # 'pi_E_2': pi_2[0],
            # 'pi_px_2': pi_2[1],
            # 'pi_py_2': pi_2[2],
            # 'pi_pz_2': pi_2[3],
            # 'pi0_E_1': pi0_1[0],
            # 'pi0_px_1': pi0_1[1],
            # 'pi0_py_1': pi0_1[2],
            # 'pi0_pz_1': pi0_1[3],
            # 'pi0_E_2': pi0_2[0],
            # 'pi0_px_2': pi0_2[1],
            # 'pi0_py_2': pi0_2[2],
            # 'pi0_pz_2': pi0_2[3],
        }
        # additional info from .root
        if not self.gen:
            df_inputs_data.update({
                'aco_angle_1': df['aco_angle_1'],
                'aco_angle_5': df['aco_angle_5'],
                'aco_angle_6': df['aco_angle_6'],
                'aco_angle_7': df['aco_angle_7'],
                'y_1_1': df['y_1_1'],
                'y_1_2': df['y_1_2'],
                'ip_x_1': df['ip_x_1'],
                'ip_y_1': df['ip_y_1'],
                'ip_z_1': df['ip_z_1'],
                'ip_x_2': df['ip_x_2'],
                'ip_y_2': df['ip_y_2'],
                'ip_z_2': df['ip_z_2'],
            })
        else:
            df_inputs_data.update({
                'sv_x_1': df['sv_x_1'],
                'sv_y_1': df['sv_y_1'],
                'sv_z_1': df['sv_z_1'],
                'sv_x_2': df['sv_x_2'],
                'sv_y_2': df['sv_y_2'],
                'sv_z_2': df['sv_z_2'],
            })
        return df_inputs_data

    def rotateVectors(self, **kwargs):
        """
        Rotates the four vectors in boosted frame
        All kwarg 4 vectors must be boosted!
        """
        print(f'Rotating in {self.channel} channel')
        if self.channel == 'rho_rho':
            pi_1_boosted = kwargs['pi_1']
            pi_2_boosted = kwargs['pi_2']
            pi0_1_boosted = kwargs['pi0_1']
            pi0_2_boosted = kwargs['pi0_2']
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted + pi0_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            pi_1_boosted_rot = np.einsum('ij,ikj->ik', pi_1_boosted[1:].T, self.rotationMatrices)
            pi0_1_boosted_rot = np.einsum('ij,ikj->ik', pi0_1_boosted[1:].T, self.rotationMatrices)
            pi_2_boosted_rot = np.einsum('ij,ikj->ik', pi_2_boosted[1:].T, self.rotationMatrices)
            pi0_2_boosted_rot = np.einsum('ij,ikj->ik', pi0_2_boosted[1:].T, self.rotationMatrices)
            return pi_1_boosted_rot, pi0_1_boosted_rot, pi_2_boosted_rot, pi0_2_boosted_rot
        elif self.channel == 'rho_a1':
            # rho is aligned to +ve z
            pi_1_boosted = kwargs['pi_1']
            pi0_1_boosted = kwargs['pi0_1']
            pi_2_boosted = kwargs['pi_2']
            pi2_2_boosted = kwargs['pi2_2']
            pi3_2_boosted = kwargs['pi3_2']
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted + pi0_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            pi_1_boosted_rot = np.einsum('ij,ikj->ik', pi_1_boosted[1:].T, self.rotationMatrices)
            pi0_1_boosted_rot = np.einsum('ij,ikj->ik', pi0_1_boosted[1:].T, self.rotationMatrices)
            pi_2_boosted_rot = np.einsum('ij,ikj->ik', pi_2_boosted[1:].T, self.rotationMatrices)
            pi2_2_boosted_rot = np.einsum('ij,ikj->ik', pi2_2_boosted[1:].T, self.rotationMatrices)
            pi3_2_boosted_rot = np.einsum('ij,ikj->ik', pi3_2_boosted[1:].T, self.rotationMatrices)
            return pi_1_boosted_rot, pi0_1_boosted_rot, pi_2_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot
        elif self.channel == 'a1_a1':
            pi_1_boosted = kwargs['pi_1']
            pi2_1_boosted = kwargs['pi2_1']
            pi3_1_boosted = kwargs['pi3_1']
            pi_2_boosted = kwargs['pi_2']
            pi2_2_boosted = kwargs['pi2_2']
            pi3_2_boosted = kwargs['pi3_2']
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted+pi2_1_boosted+pi3_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            pi_1_boosted_rot = np.einsum('ij,ikj->ik', pi_1_boosted[1:].T, self.rotationMatrices)
            pi2_1_boosted_rot = np.einsum('ij,ikj->ik', pi2_1_boosted[1:].T, self.rotationMatrices)
            pi3_1_boosted_rot = np.einsum('ij,ikj->ik', pi3_1_boosted[1:].T, self.rotationMatrices)
            pi_2_boosted_rot = np.einsum('ij,ikj->ik', pi_2_boosted[1:].T, self.rotationMatrices)
            pi2_2_boosted_rot = np.einsum('ij,ikj->ik', pi2_2_boosted[1:].T, self.rotationMatrices)
            pi3_2_boosted_rot = np.einsum('ij,ikj->ik', pi3_2_boosted[1:].T, self.rotationMatrices)
            return pi_1_boosted_rot, pi2_1_boosted_rot, pi3_1_boosted_rot, pi_2_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot
        else:
            raise ValueError('Channel input not understood')

    def rotationMatrixVectorised(self, vec1, vec2):
        """
        Find the rotation matrix that aligns vec1 to vec2
        Expects vec1, vec2 to be list of vectors
        Returns list of matrices
        """
        a, b = vec1 / np.linalg.norm(vec1, axis=1)[:, None], vec2 / np.linalg.norm(vec2, axis=1)[:, None]
        v = np.cross(a, b)
        c = np.einsum('ij, ij->i', a, b)
        s = np.linalg.norm(v, axis=1)
        kmat = np.array([[-np.zeros(len(vec1)), v.T[2], -v.T[1]], [-v.T[2], -np.zeros(len(vec1)), v.T[0]], [v.T[1], -v.T[0], -np.zeros(len(vec1))]]).T
        rotation_matrix = np.tile(np.eye(3), (len(vec1), 1, 1)) + kmat + np.linalg.matrix_power(kmat, 2)*((1 - c) / (s ** 2))[:, None][:, np.newaxis]
        return rotation_matrix

    def getAcoAngles(self, **kwargs):
        """
        Returns all the aco angles for different channels
        """
        if self.channel == 'rho_rho':
            pi_1 = kwargs['pi_1']
            pi_2 = kwargs['pi_2']
            pi0_1 = kwargs['pi0_1']
            pi0_2 = kwargs['pi0_2']
            y_rho_1, y_rho_2 = self.getY(pi_1=pi_1, pi_2=pi_2, pi0_1=pi0_1, pi0_2=pi0_2)
            zmf = pi_1 + pi_2 + pi0_1 + pi0_2
            aco_angle_1 = self.getAcoAnglesForOneRF(pi0_1, pi0_2, pi_1, pi_2, zmf, y_rho_1, y_rho_2)
            # should be no nans so can delete next 2 lines?
            # print('number of nans using perp calculation:', np.sum(np.isnan(aco_angle_1)))
            aco_angle_1[np.isnan(aco_angle_1)] = np.pi
            return aco_angle_1
        elif self.channel == 'rho_a1':
            # 4 aco angles
            pi_1 = kwargs['pi_1']
            pi0_1 = kwargs['pi0_1']
            rho_1 = pi_1 + pi0_1
            pi_2 = kwargs['pi_2']
            pi2_2 = kwargs['pi2_2']
            pi3_2 = kwargs['pi3_2']
            # rho +/- , rho 0 frame
            zmf_1 = rho_1 + pi_2 + pi2_2
            zmf_2 = rho_1 + pi_2 + pi3_2
            y_rho_1, y_rho0_2, y_rho02_2, y_a1_2, y_a12_2 = self.getY(pi_1=pi_1, pi0_1=pi0_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
            aco_angle_1 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi0_1, pi2_2, zmf_1, y_rho_1, y_rho0_2)
            aco_angle_2 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi0_1, pi3_2, zmf_2, y_rho_1, y_rho02_2)
            # rho +/-, a1 frame
            # a1 -> rho0, pi +/-
            zmf_3 = rho_1 + pi_2 + pi2_2 + pi3_2
            aco_angle_3 = self.getAcoAnglesForOneRF(pi_1, pi_2 + pi2_2, pi0_1, pi3_2, zmf_3, y_rho_1, y_a1_2)
            aco_angle_4 = self.getAcoAnglesForOneRF(pi_1, pi_2 + pi3_2, pi0_1, pi2_2, zmf_3, y_rho_1, y_a12_2)
            return aco_angle_1, aco_angle_2, aco_angle_3, aco_angle_4
        elif self.channel == 'a1_a1':
            # 16 aco angles
            pi_1 = kwargs['pi_1']
            pi2_1 = kwargs['pi2_1']
            pi3_1 = kwargs['pi3_1']
            pi_2 = kwargs['pi_2']
            pi2_2 = kwargs['pi2_2']
            pi3_2 = kwargs['pi3_2']
            a1_1 = pi_1 + pi2_1 + pi3_1
            a1_2 = pi_2 + pi2_2 + pi3_2
            y_rho0_1, y_rho02_1, y_rho0_2, y_rho02_2, y_a1_1, y_a12_1, y_a1_2, y_a12_2 = self.getY(pi_1=pi_1, pi2_1=pi2_1, pi3_1=pi3_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
            # rho0, rho0 frame
            zmf_1 = pi_1 + pi2_1 + pi_2 + pi2_2
            zmf_2 = pi_1 + pi3_1 + pi_2 + pi2_2
            zmf_3 = pi_1 + pi2_1 + pi_2 + pi3_2
            zmf_4 = pi_1 + pi3_1 + pi_2 + pi3_2
            aco_angle_1 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi2_1, pi2_2, zmf_1, y_rho0_1, y_rho0_2)
            aco_angle_2 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi3_1, pi2_2, zmf_2, y_rho02_1, y_rho0_2)
            aco_angle_3 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi2_1, pi3_2, zmf_3, y_rho0_1, y_rho02_2)
            aco_angle_4 = self.getAcoAnglesForOneRF(pi_1, pi_2, pi3_1, pi3_2, zmf_4, y_rho02_1, y_rho02_2)
            # rho0, a1 frame
            zmf_5 = pi_1 + pi2_1 + a1_2
            zmf_6 = pi_1 + pi3_1 + a1_2
            aco_angle_5 = self.getAcoAnglesForOneRF(pi_1, pi_2+pi2_2, pi2_1, pi3_2, zmf_5, y_rho0_1, y_a1_2)
            aco_angle_6 = self.getAcoAnglesForOneRF(pi_1, pi_2+pi3_2, pi2_1, pi2_2, zmf_5, y_rho0_1, y_a12_2)
            aco_angle_7 = self.getAcoAnglesForOneRF(pi_1, pi_2+pi2_2, pi3_1, pi3_2, zmf_6, y_rho02_1, y_a1_2)
            aco_angle_8 = self.getAcoAnglesForOneRF(pi_1, pi_2+pi3_2, pi3_1, pi2_2, zmf_6, y_rho02_1, y_a12_2)
            # a1, rho0 frame
            zmf_7 = a1_1 + pi_2 + pi2_2
            zmf_8 = a1_1 + pi_2 + pi3_2
            aco_angle_9 = self.getAcoAnglesForOneRF(pi_1+pi2_1, pi_2, pi3_1, pi2_2, zmf_7, y_a1_1, y_rho0_2)
            aco_angle_10 = self.getAcoAnglesForOneRF(pi_1+pi3_1, pi_2, pi2_1, pi2_2, zmf_7, y_a12_1, y_rho0_2)
            aco_angle_11 = self.getAcoAnglesForOneRF(pi_1+pi2_1, pi_2, pi3_1, pi3_2, zmf_8, y_a1_1, y_rho02_2)
            aco_angle_12 = self.getAcoAnglesForOneRF(pi_1+pi3_1, pi_2, pi2_1, pi3_2, zmf_8, y_a12_1, y_rho02_2)
            # a1, a1 frame
            zmf_9 = a1_1 + a1_2
            aco_angle_13 = self.getAcoAnglesForOneRF(pi_1+pi2_1, pi_2+pi2_2, pi3_1, pi3_2, zmf_9, y_a1_1, y_a1_2)
            aco_angle_14 = self.getAcoAnglesForOneRF(pi_1+pi3_1, pi_2+pi2_2, pi2_1, pi3_2, zmf_9, y_a12_1, y_a1_2)
            aco_angle_15 = self.getAcoAnglesForOneRF(pi_1+pi2_1, pi_2+pi3_2, pi3_1, pi2_2, zmf_9, y_a1_1, y_a12_2)
            aco_angle_16 = self.getAcoAnglesForOneRF(pi_1+pi3_1, pi_2+pi3_2, pi2_1, pi2_2, zmf_9, y_a12_1, y_a12_2)
            return aco_angle_1, aco_angle_2, aco_angle_3, aco_angle_4, aco_angle_5, aco_angle_6, aco_angle_7, aco_angle_8, aco_angle_9, aco_angle_10, aco_angle_11, aco_angle_12, aco_angle_13, aco_angle_14, aco_angle_15, aco_angle_16
        else:
            raise ValueError('Channel not understood')

    def getAcoAnglesForOneRF(self, p1, p2, p3, p4, rest_frame, y_1_1=None, y_1_2=None):
        """
        Calculates aco angles for a rest frame (rest_frame)
        p1, p3 are pairs from same decay
        p2, p4 are pairs from same decay
        """
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        p1 = p1.boost_particle(boost)
        p2 = p2.boost_particle(boost)
        p3 = p3.boost_particle(boost)
        p4 = p4.boost_particle(boost)
        # Some geometrical functions

        def cross_product(vector3_1, vector3_2):
            return np.cross(vector3_1.T, vector3_2.T).T

        def dot_product(vector1, vector2):
            return np.einsum('ij, ij->i', vector1.T, vector2.T)

        def norm(vector):
            return np.sqrt((vector.T ** 2).sum(-1))[..., np.newaxis].T
        # calculating the perpependicular component
        pi0_1_3Mom_star_perp = cross_product(p1[1:], p3[1:])
        pi0_2_3Mom_star_perp = cross_product(p2[1:], p4[1:])
        # Now normalise:
        pi0_1_3Mom_star_perp = pi0_1_3Mom_star_perp/norm(pi0_1_3Mom_star_perp)
        pi0_2_3Mom_star_perp = pi0_2_3Mom_star_perp/norm(pi0_2_3Mom_star_perp)
        # Calculating phi_star
        phi_CP = np.arccos(dot_product(pi0_1_3Mom_star_perp, pi0_2_3Mom_star_perp))
        if y_1_1 is not None:
            # The O variable
            cross = np.cross(pi0_1_3Mom_star_perp.transpose(), pi0_2_3Mom_star_perp.transpose()).transpose()
            bigO = dot_product(p4[1:], cross)
            # The energy ratios
            y_T = np.array(y_1_1 * y_1_2)
            # perform the shift w.r.t. O* sign
            phi_CP = np.where(bigO >= 0, 2*np.pi-phi_CP, phi_CP)
            # additionnal shift that needs to be done do see differences between odd and even scenarios, with y=Energy ratios
            phi_CP = np.where(y_T >= 0, np.where(phi_CP < np.pi, phi_CP+np.pi, phi_CP-np.pi), phi_CP)
        return phi_CP

    def getY(self, **kwargs):
        """
        Calculates y for given decay
        4 vectors are NOT boosted - calculating in lab frame
        """
        if self.channel == 'rho_rho':
            pi_1 = kwargs['pi_1']
            pi_2 = kwargs['pi_2']
            pi0_1 = kwargs['pi0_1']
            pi0_2 = kwargs['pi0_2']
            y_rho_1 = (pi_1.e - pi0_1.e)/(pi_1.e + pi0_1.e)
            y_rho_2 = (pi_2.e - pi0_2.e)/(pi_2.e + pi0_2.e)
            return y_rho_1, y_rho_2
        elif self.channel == 'rho_a1':
            # 5 ys
            pi_1 = kwargs['pi_1']
            pi0_1 = kwargs['pi0_1']
            rho_1 = pi_1 + pi0_1
            pi_2 = kwargs['pi_2']
            pi2_2 = kwargs['pi2_2']
            pi3_2 = kwargs['pi3_2']
            # 1 y from equation 1
            y_rho_1 = (pi_1.e - pi0_1.e)/(pi_1.e + pi0_1.e)
            # from the y_rho0 part, 2 values due to ambiguity: rho0 can either be pi_2+pi2_2 or pi_2+pi3_2
            y_rho0_2 = (pi_2.e - pi2_2.e)/(pi_2.e + pi2_2.e)
            y_rho02_2 = (pi_2.e - pi3_2.e)/(pi_2.e + pi3_2.e)
            # from y_a1 part, 2 values due to ambiguity
            rho0_2 = pi_2 + pi2_2
            a1 = rho0_2 + pi3_2
            y_a1_2 = (rho0_2.e - pi3_2.e) / (rho0_2.e + pi3_2.e) - (self.getMSquared(a1) - self.getMSquared(pi3_2) + self.getMSquared(rho0_2)) / (2 * self.getMSquared(a1))
            rho02_2 = pi_2 + pi3_2
            y_a12_2 = (rho02_2.e - pi2_2.e) / (rho02_2.e + pi2_2.e) - (self.getMSquared(a1) - self.getMSquared(pi2_2) + self.getMSquared(rho02_2)) / (2 * self.getMSquared(a1))
            return y_rho_1, y_rho0_2, y_rho02_2, y_a1_2, y_a12_2
        elif self.channel == 'a1_a1':
            # 8 ys
            pi_1 = kwargs['pi_1']
            pi2_1 = kwargs['pi2_1']
            pi3_1 = kwargs['pi3_1']
            pi_2 = kwargs['pi_2']
            pi2_2 = kwargs['pi2_2']
            pi3_2 = kwargs['pi3_2']
            rho0_1 = pi_1 + pi2_1
            rho02_1 = pi_1 + pi3_1
            rho0_2 = pi_2 + pi2_2
            rho02_2 = pi_2 + pi3_2
            a1_1 = rho0_1 + pi3_1
            a1_2 = rho0_2 + pi3_2
            # 4 ys from the y_a1 formula due to ambiguities in the 2 a1s
            # 2 from the first a1
            y_a1_1 = (rho0_1.e - pi3_1.e) - (a1_1.m**2 - pi3_1.m**2 + rho0_1.m) / (2 * a1_1.m**2)
            y_a12_1 = (rho02_1.e - pi2_1.e) - (a1_1.m**2 - pi2_1.m**2 + rho02_1.m) / (2 * a1_1.m**2)
            # 2 from the second a1
            y_a1_2 = (rho0_2.e - pi3_2.e) - (a1_2.m**2 - pi3_2.m**2 + rho0_2.m) / (2 * a1_2.m**2)
            y_a12_2 = (rho02_2.e - pi2_2.e) - (a1_2.m**2 - pi2_2.m**2 + rho02_2.m) / (2 * a1_2.m**2)
            y_a1_1 = (rho0_1.e - pi3_1.e) / (rho0_1.e + pi3_1.e) - (self.getMSquared(a1_1) - self.getMSquared(pi3_1) + self.getMSquared(rho0_1)) / (2 * self.getMSquared(a1_1))
            y_a12_1 = (rho02_1.e - pi2_1.e) / (rho02_1.e + pi2_1.e) - (self.getMSquared(a1_1) - self.getMSquared(pi2_1) + self.getMSquared(rho02_1)) / (2 * self.getMSquared(a1_1))
            # 2 from the second a1
            y_a1_2 = (rho0_2.e - pi3_2.e) / (rho0_2.e + pi3_2.e) - (self.getMSquared(a1_2) - self.getMSquared(pi3_2) + self.getMSquared(rho0_2)) / (2 * self.getMSquared(a1_2))
            y_a12_2 = (rho02_2.e - pi2_2.e) / (rho02_2.e + pi2_2.e) - (self.getMSquared(a1_2) - self.getMSquared(pi2_2) + self.getMSquared(rho02_2)) / (2 * self.getMSquared(a1_2))
            # 4 ys from the y_rho0 due to ambiguities in the 2 rho0s
            # 2 from the first rho0
            y_rho0_1 = (pi_1.e - pi2_1.e) / (pi_1.e + pi2_1.e)
            y_rho02_1 = (pi_1.e - pi3_1.e) / (pi_1.e + pi3_1.e)
            # 2 from the second rho0
            y_rho0_2 = (pi_2.e - pi2_2.e) / (pi_2.e + pi2_2.e)
            y_rho02_2 = (pi_2.e - pi3_2.e) / (pi_2.e + pi3_2.e)
            return y_rho0_1, y_rho02_1, y_rho0_2, y_rho02_2, y_a1_1, y_a12_1, y_a1_2, y_a12_2
        else:
            raise ValueError('Channel not understood')

    def getMSquared(self, four_vector: Momentum4):
        return four_vector[0]**2-(four_vector[1]**2+four_vector[2]**2+four_vector[3]**2)

    def calculateRhoA1Data(self, df):
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
        pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
        rho_1 = pi_1 + pi0_1  # charged rho
        rho0_2 = pi_2 + pi2_2
        rho02_2 = pi_2 + pi3_2
        a1_2 = pi_2 + pi2_2 + pi3_2
        rest_frame = rho_1 + a1_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi2_2_boosted = pi2_2.boost_particle(boost)
        pi3_2_boosted = pi3_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho0_2_boosted = pi_2_boosted + pi2_2_boosted
        rho02_2_boosted = pi_2_boosted + pi3_2_boosted
        a1_2_boosted = pi_2_boosted + pi2_2_boosted + pi3_2_boosted
        br_vectors = self.rotateVectors(pi_1=pi_1_boosted, pi0_1=pi0_1_boosted, pi_2=pi_2_boosted, pi2_2=pi2_2_boosted, pi3_2=pi3_2_boosted)
        pi_1_boosted_rot, pi0_1_boosted_rot, pi_2_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot = br_vectors
        rho_1_boosted_rot = pi_1_boosted_rot + pi0_1_boosted_rot
        a1_2_boosted_rot = pi_2_boosted_rot + pi2_2_boosted_rot + pi3_2_boosted_rot
        rho0_2_boosted_rot = pi_2_boosted_rot + pi2_2_boosted_rot
        rho02_2_boosted_rot = pi_2_boosted_rot + pi3_2_boosted_rot
        aco_angle_1, aco_angle_2, aco_angle_3, aco_angle_4 = self.getAcoAngles(pi_1=pi_1, pi0_1=pi0_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
        y_rho_1, y_rho0_2, y_rho02_2, y_a1_2, y_a12_2 = self.getY(pi_1=pi_1, pi0_1=pi0_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
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
            'rho0_E_2_br': rho0_2_boosted[0],
            'rho0_px_2_br': rho0_2_boosted_rot[:, 0],
            'rho0_py_2_br': rho0_2_boosted_rot[:, 1],
            'rho0_pz_2_br': rho0_2_boosted_rot[:, 2],
            'rho02_E_2_br': rho02_2_boosted[0],
            'rho02_px_2_br': rho02_2_boosted_rot[:, 0],
            'rho02_py_2_br': rho02_2_boosted_rot[:, 1],
            'rho02_pz_2_br': rho02_2_boosted_rot[:, 2],
            'a1_E_2_br': a1_2_boosted[0],
            'a1_px_2_br': a1_2_boosted_rot[:, 0],
            'a1_py_2_br': a1_2_boosted_rot[:, 1],
            'a1_pz_2_br': a1_2_boosted_rot[:, 2],
            'aco_angle_1_calc': aco_angle_1,
            'aco_angle_2_calc': aco_angle_2,
            'aco_angle_3_calc': aco_angle_3,
            'aco_angle_4_calc': aco_angle_4,
            "y_rho_1": y_rho_1,
            "y_rho0_2": y_rho0_2,
            "y_rho02_2": y_rho02_2,
            "y_a1_2": y_a1_2,
            "y_a12_2": y_a12_2,
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_rho_1': rho_1.m,
            'm_rho0_2': rho0_2.m,
            'm_rho02_2': rho02_2.m,
            'm_a1_2': a1_2.m,
        }
        if not self.gen:
            df_inputs_data.update({
                # additional info from .root
                'aco_angle_1': df['aco_angle_1'],
                'aco_angle_2': df['aco_angle_2'],
                'aco_angle_3': df['aco_angle_3'],
                'aco_angle_4': df['aco_angle_4'],
                'y_1_1': df['y_1_1'],
                'y_1_2': df['y_1_2'],
                'y_2_2': df['y_2_2'],
                'y_3_2': df['y_3_2'],
                'y_4_2': df['y_4_2'],
                'ip_x_1': df['ip_x_1'],
                'ip_y_1': df['ip_y_1'],
                'ip_z_1': df['ip_z_1'],
                'ip_x_2': df['ip_x_2'],
                'ip_y_2': df['ip_y_2'],
                'ip_z_2': df['ip_z_2'],
                'sv_x_2': df['sv_x_2'],
                'sv_y_2': df['sv_y_2'],
                'sv_z_2': df['sv_z_2'],
            })
        else:
            df_inputs_data.update({
                'sv_x_1': df['sv_x_1'],
                'sv_y_1': df['sv_y_1'],
                'sv_z_1': df['sv_z_1'],
                'sv_x_2': df['sv_x_2'],
                'sv_y_2': df['sv_y_2'],
                'sv_z_2': df['sv_z_2'],
            })
        return df_inputs_data

    def calculateA1A1Data(self, df):
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi2_1 = Momentum4(df['pi2_E_1'], df["pi2_px_1"], df["pi2_py_1"], df["pi2_pz_1"])
        pi3_1 = Momentum4(df['pi3_E_1'], df["pi3_px_1"], df["pi3_py_1"], df["pi3_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
        pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
        a1_1 = pi_1 + pi2_1 + pi3_1
        a1_2 = pi_2 + pi2_2 + pi3_2
        rho0_1 = pi_1 + pi2_1
        rho02_1 = pi_1 + pi3_1
        rho0_2 = pi_2 + pi2_2
        rho02_2 = pi_2 + pi3_2
        rest_frame = a1_1 + a1_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi2_1_boosted = pi2_1.boost_particle(boost)
        pi3_1_boosted = pi3_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi2_2_boosted = pi2_2.boost_particle(boost)
        pi3_2_boosted = pi3_2.boost_particle(boost)
        rho0_1_boosted = pi_1_boosted + pi2_1_boosted
        rho02_1_boosted = pi_1_boosted + pi3_1_boosted
        rho0_2_boosted = pi_2_boosted + pi2_2_boosted
        rho02_2_boosted = pi_2_boosted + pi3_2_boosted
        a1_1_boosted = pi_1_boosted + pi2_1_boosted + pi3_1_boosted
        a1_2_boosted = pi_2_boosted + pi2_2_boosted + pi3_2_boosted
        br_vectors = self.rotateVectors(pi_1=pi_1_boosted, pi2_1=pi2_1_boosted, pi3_1=pi3_1_boosted, pi_2=pi_2_boosted, pi2_2=pi2_2_boosted, pi3_2=pi3_2_boosted)
        pi_1_boosted_rot, pi2_1_boosted_rot, pi3_1_boosted_rot, pi_2_boosted_rot, pi2_2_boosted_rot, pi3_2_boosted_rot = br_vectors
        rho0_1_boosted_rot = pi_1_boosted_rot + pi2_1_boosted_rot
        rho02_1_boosted_rot = pi_1_boosted_rot + pi3_1_boosted_rot
        rho0_2_boosted_rot = pi_2_boosted_rot + pi2_2_boosted_rot
        rho02_2_boosted_rot = pi_2_boosted_rot + pi3_2_boosted_rot
        a1_1_boosted_rot = pi_1_boosted_rot + pi2_1_boosted_rot + pi3_1_boosted_rot
        a1_2_boosted_rot = pi_2_boosted_rot + pi2_2_boosted_rot + pi3_2_boosted_rot
        aco_angles = self.getAcoAngles(pi_1=pi_1, pi2_1=pi2_1, pi3_1=pi3_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
        aco_angle_1, aco_angle_2, aco_angle_3, aco_angle_4, aco_angle_5, aco_angle_6, aco_angle_7, aco_angle_8, aco_angle_9, aco_angle_10, aco_angle_11, aco_angle_12, aco_angle_13, aco_angle_14, aco_angle_15, aco_angle_16 = aco_angles
        y_rho0_1, y_rho02_1, y_rho_2, y_rho0_2, y_a1_1, y_a12_1, y_a1_2, y_a12_2 = self.getY(pi_1=pi_1, pi2_1=pi2_1, pi3_1=pi3_1, pi_2=pi_2, pi2_2=pi2_2, pi3_2=pi3_2)
        df_inputs_data = {
            'pi_E_1_br': pi_1_boosted[0],
            'pi_px_1_br': pi_1_boosted_rot[:, 0],
            'pi_py_1_br': pi_1_boosted_rot[:, 1],
            'pi_pz_1_br': pi_1_boosted_rot[:, 2],
            'pi2_E_1_br': pi2_1_boosted[0],
            'pi2_px_1_br': pi2_1_boosted_rot[:, 0],
            'pi2_py_1_br': pi2_1_boosted_rot[:, 1],
            'pi2_pz_1_br': pi2_1_boosted_rot[:, 2],
            'pi3_E_1_br': pi3_1_boosted[0],
            'pi3_px_1_br': pi3_1_boosted_rot[:, 0],
            'pi3_py_1_br': pi3_1_boosted_rot[:, 1],
            'pi3_pz_1_br': pi3_1_boosted_rot[:, 2],
            'pi_E_2_br': pi_2_boosted[0],
            'pi_px_2_br': pi_2_boosted_rot[:, 0],
            'pi_py_2_br': pi_2_boosted_rot[:, 1],
            'pi_pz_2_br': pi_2_boosted_rot[:, 2],
            'pi2_E_2_br': pi2_2_boosted[0],
            'pi2_px_2_br': pi2_2_boosted_rot[:, 0],
            'pi2_py_2_br': pi2_2_boosted_rot[:, 1],
            'pi2_pz_2_br': pi2_2_boosted_rot[:, 2],
            'pi3_E_2_br': pi3_2_boosted[0],
            'pi3_px_2_br': pi3_2_boosted_rot[:, 0],
            'pi3_py_2_br': pi3_2_boosted_rot[:, 1],
            'pi3_pz_2_br': pi3_2_boosted_rot[:, 2],
            'rho0_E_1_br': rho0_1_boosted[0],
            'rho0_px_1_br': rho0_1_boosted_rot[:, 0],
            'rho0_py_1_br': rho0_1_boosted_rot[:, 1],
            'rho0_pz_1_br': rho0_1_boosted_rot[:, 2],
            'rho02_E_1_br': rho02_1_boosted[0],
            'rho02_px_1_br': rho02_1_boosted_rot[:, 0],
            'rho02_py_1_br': rho02_1_boosted_rot[:, 1],
            'rho02_pz_1_br': rho02_1_boosted_rot[:, 2],
            'rho0_E_2_br': rho0_2_boosted[0],
            'rho0_px_2_br': rho0_2_boosted_rot[:, 0],
            'rho0_py_2_br': rho0_2_boosted_rot[:, 1],
            'rho0_pz_2_br': rho0_2_boosted_rot[:, 2],
            'rho02_E_2_br': rho02_2_boosted[0],
            'rho02_px_2_br': rho02_2_boosted_rot[:, 0],
            'rho02_py_2_br': rho02_2_boosted_rot[:, 1],
            'rho02_pz_2_br': rho02_2_boosted_rot[:, 2],
            'a1_E_1_br': a1_1_boosted[0],
            'a1_px_1_br': a1_1_boosted_rot[:, 0],
            'a1_py_1_br': a1_1_boosted_rot[:, 1],
            'a1_pz_1_br': a1_1_boosted_rot[:, 2],
            'a1_E_2_br': a1_2_boosted[0],
            'a1_px_2_br': a1_2_boosted_rot[:, 0],
            'a1_py_2_br': a1_2_boosted_rot[:, 1],
            'a1_pz_2_br': a1_2_boosted_rot[:, 2],
            'aco_angle_1_calc': aco_angle_1,
            'aco_angle_2_calc': aco_angle_2,
            'aco_angle_3_calc': aco_angle_3,
            'aco_angle_4_calc': aco_angle_4,
            'aco_angle_5_calc': aco_angle_5,
            'aco_angle_6_calc': aco_angle_6,
            'aco_angle_7_calc': aco_angle_7,
            'aco_angle_8_calc': aco_angle_8,
            'aco_angle_9_calc': aco_angle_9,
            'aco_angle_10_calc': aco_angle_10,
            'aco_angle_11_calc': aco_angle_11,
            'aco_angle_12_calc': aco_angle_12,
            'aco_angle_13_calc': aco_angle_13,
            'aco_angle_14_calc': aco_angle_14,
            'aco_angle_15_calc': aco_angle_15,
            'aco_angle_16_calc': aco_angle_16,
            "y_rho0_1": y_rho0_1,
            "y_rho02_1": y_rho02_1,
            "y_rho_2": y_rho_2,
            "y_rho0_2": y_rho0_2,
            "y_a1_1": y_a1_1,
            "y_a12_1": y_a12_1,
            "y_a1_2": y_a1_2,
            "y_a12_2": y_a12_2,
            'w_a': df.wt_cp_sm,
            'w_b': df.wt_cp_ps,
            'm_rho0_2': rho0_1.m,
            'm_rho02_2': rho02_1.m,
            'm_rho0_2': rho0_2.m,
            'm_rho02_2': rho02_2.m,
            'm_a1_1': a1_1.m,
            'm_a1_1': a1_2.m,
            # additional info from .root
        }
        if not self.gen:
            df_inputs_data.update({
                'aco_angle_1': df['aco_angle_1'],
                'aco_angle_2': df['aco_angle_2'],
                'aco_angle_3': df['aco_angle_3'],
                'aco_angle_4': df['aco_angle_4'],
                'y_1_1': df['y_1_1'],
                'y_1_2': df['y_1_2'],
                'y_2_2': df['y_2_2'],
                'y_3_2': df['y_3_2'],
                'y_4_2': df['y_4_2'],
                'ip_x_1': df['ip_x_1'],
                'ip_y_1': df['ip_y_1'],
                'ip_z_1': df['ip_z_1'],
                'ip_x_2': df['ip_x_2'],
                'ip_y_2': df['ip_y_2'],
                'ip_z_2': df['ip_z_2'],
                'sv_x_1': df['sv_x_1'],
                'sv_y_1': df['sv_y_1'],
                'sv_z_1': df['sv_z_1'],
                'sv_x_2': df['sv_x_2'],
                'sv_y_2': df['sv_y_2'],
                'sv_z_2': df['sv_z_2'],
                'pv_angle': df['pv_angle'],
            })
        else:
            df_inputs_data.update({
                'sv_x_1': df['sv_x_1'],
                'sv_y_1': df['sv_y_1'],
                'sv_z_1': df['sv_z_1'],
                'sv_x_2': df['sv_x_2'],
                'sv_y_2': df['sv_y_2'],
                'sv_z_2': df['sv_z_2'],
            })
        return df_inputs_data

    def createAddons(self, addons, df, df_inputs, binary, addons_config={}):
        """
        If you want to create more addon features, put the necessary arguments through kwargs,
        unpack them at the start of this function, and add an if case to your needs
        TODO: need to catch incorrectly loaded kwargs
        Return: df_inputs (modified)
        """
        for addon in addons:
            if addon == 'met':
                print('Addon MET loaded')
                boost = self.createBoostAndRotationMatrices(df)
                metx_b, mety_b = self.addonMET(df, boost)
                # TODO: CHANGE
                df_inputs['metx_b'] = metx_b
                df_inputs['mety_b'] = mety_b
            if addon == 'neutrino':
                print('Addon neutrino loaded')
                load_alpha = addons_config['neutrino']['load_alpha']
                termination = addons_config['neutrino']['termination']
                imputer_mode = addons_config['neutrino']['imputer_mode']
                # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = self.addonNeutrinos(df, df_inputs, binary, load_alpha, termination=termination)
                # df_inputs['alpha_1'] = alpha_1
                # df_inputs['alpha_2'] = alpha_2
                # df_inputs['E_nu_1'] = E_nu_1
                # df_inputs['E_nu_2'] = E_nu_2
                # df_inputs['p_t_nu_1'] = p_t_nu_1
                # df_inputs['p_t_nu_2'] = p_t_nu_2
                # df_inputs['p_z_nu_1'] = p_z_nu_1
                # df_inputs['p_z_nu_2'] = p_z_nu_2
                if imputer_mode == 'remove':
                    # modifies the original df by removing events
                    df_inputs, df = self.addonNeutrinos(df, df_inputs, binary, load_alpha, imputer_mode, termination=termination)
                else:
                    df_inputs = self.addonNeutrinos(df, df_inputs, binary, load_alpha, imputer_mode, termination=termination)
            if addon == 'ip':
                print('Impact paramter loaded')
                boost = self.createBoostAndRotationMatrices(df)
                ip_1_boosted_rot, ip_2_boosted_rot = self.addonIP(df, boost)
                df_inputs['ip_x_1_br'] = ip_1_boosted_rot[:, 0]
                df_inputs['ip_y_1_br'] = ip_1_boosted_rot[:, 1]
                df_inputs['ip_z_1_br'] = ip_1_boosted_rot[:, 2]
                df_inputs['ip_x_2_br'] = ip_2_boosted_rot[:, 0]
                df_inputs['ip_y_2_br'] = ip_2_boosted_rot[:, 1]
                df_inputs['ip_z_2_br'] = ip_2_boosted_rot[:, 2]
            if addon == 'sv':
                boost = self.createBoostAndRotationMatrices(df)
                if self.channel == 'a1_a1' or self.gen:
                    sv_1_boosted_rot, sv_2_boosted_rot = self.addonSV(df, boost)
                    df_inputs['sv_x_1_br'] = sv_1_boosted_rot[:, 0]
                    df_inputs['sv_y_1_br'] = sv_1_boosted_rot[:, 1]
                    df_inputs['sv_z_1_br'] = sv_1_boosted_rot[:, 2]
                    df_inputs['sv_x_2_br'] = sv_2_boosted_rot[:, 0]
                    df_inputs['sv_y_2_br'] = sv_2_boosted_rot[:, 1]
                    df_inputs['sv_z_2_br'] = sv_2_boosted_rot[:, 2]
                elif self.channel == 'rho_a1':
                    sv_2_boosted_rot = self.addonSV(df, boost)
                    df_inputs['sv_x_2_br'] = sv_2_boosted_rot[:, 0]
                    df_inputs['sv_y_2_br'] = sv_2_boosted_rot[:, 1]
                    df_inputs['sv_z_2_br'] = sv_2_boosted_rot[:, 2]
                else:
                    print('WARNING (createAddons): No SV on rho_rho channel!')

        return df_inputs

    def addonMET(self, df, boost):
        """
        Addon configuration for the MET
        """
        N = len(df['metx'])
        met_x = Momentum4(df['metx'], np.zeros(N), np.zeros(N), np.zeros(N))
        met_y = Momentum4(df['mety'], np.zeros(N), np.zeros(N), np.zeros(N))
        # boost MET
        metx_b = met_x.boost_particle(boost)[0]
        mety_b = met_y.boost_particle(boost)[0]
        return metx_b, mety_b

    def addonNeutrinos(self, df, df_inputs, binary, load_alpha, imputer_mode, termination=100):
        """
        Addon configuration for neutrino information
        TODO:
        - load in neutrino phis
        -- Returns: alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 --
        Returns: df_inputs (modified)
        """
        NR = NeutrinoReconstructor(binary, self.channel)
        if not self.gen:
            df_inputs = NR.runAlphaReconstructor(df.reset_index(drop=True), df_inputs.reset_index(drop=True), load_alpha=load_alpha, termination=termination)
        else:
            df_inputs = NR.runGenAlphaReconstructor(df.reset_index(drop=True), df_inputs.reset_index(drop=True), load_alpha=load_alpha)
        if imputer_mode == 'remove':
            df_inputs_imputed, df = NR.dealWithMissingData(df_inputs, imputer_mode, df=df)
            return df_inputs_imputed, df
        # return df_inputs_imputed
        df_inputs_imputed = NR.dealWithMissingData(df_inputs, imputer_mode)
        return df_inputs_imputed

    def addonIP(self, df, boost):
        N = len(df.ip_x_1)
        ip_1 = Momentum4(np.zeros(N), df.ip_x_1, df.ip_y_1, df.ip_z_1)
        ip_2 = Momentum4(np.zeros(N), df.ip_x_2, df.ip_y_2, df.ip_z_2)
        ip_1_boosted = ip_1.boost_particle(boost)
        ip_2_boosted = ip_2.boost_particle(boost)
        ip_1_boosted_rot = np.einsum('ij,ikj->ik', ip_1_boosted[1:].T, self.rotationMatrices)
        ip_2_boosted_rot = np.einsum('ij,ikj->ik', ip_2_boosted[1:].T, self.rotationMatrices)

        return ip_1_boosted_rot, ip_2_boosted_rot

    def addonSV(self, df, boost):
        """
        Do not call on rho_rho channel
        """
        N = len(df.sv_x_2)
        if self.channel == 'a1_a1' or self.gen:
            sv_1 = Momentum4(np.zeros(N), df.sv_x_1, df.sv_y_1, df.sv_z_1)
            sv_2 = Momentum4(np.zeros(N), df.sv_x_2, df.sv_y_2, df.sv_z_2)
            sv_1_boosted = sv_1.boost_particle(boost)
            sv_2_boosted = sv_2.boost_particle(boost)
            sv_1_boosted_rot = np.einsum('ij,ikj->ik', sv_1_boosted[1:].T, self.rotationMatrices)
            sv_2_boosted_rot = np.einsum('ij,ikj->ik', sv_2_boosted[1:].T, self.rotationMatrices)
            return sv_1_boosted_rot, sv_2_boosted_rot
        elif self.channel == 'rho_a1':
            sv_2 = Momentum4(np.zeros(N), df.sv_x_2, df.sv_y_2, df.sv_z_2)
            sv_2_boosted = sv_2.boost_particle(boost)
            sv_2_boosted_rot = np.einsum('ij,ikj->ik', sv_2_boosted[1:].T, self.rotationMatrices)
            return sv_2_boosted_rot
        else:
            raise ValueError('No SV on rho_rho channel/ Channel not understood')

    def createBoostAndRotationMatrices(self, df):
        if self.channel == 'rho_rho':
            pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
            pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
            pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
            pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
            rho_1 = pi_1 + pi0_1
            rho_2 = pi_2 + pi0_2
            rest_frame = rho_1 + rho_2
            boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
            pi_1_boosted = pi_1.boost_particle(boost)
            pi0_1_boosted = pi0_1.boost_particle(boost)
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted + pi0_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            return boost
        elif self.channel == 'rho_a1':
            pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
            pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
            pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
            pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
            pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
            rho_1 = pi_1 + pi0_1  # charged rho
            a1_2 = pi_2 + pi2_2 + pi3_2
            rest_frame = rho_1 + a1_2
            boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
            pi_1_boosted = pi_1.boost_particle(boost)
            pi0_1_boosted = pi0_1.boost_particle(boost)
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted + pi0_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            return boost
        elif self.channel == 'a1_a1':
            pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
            pi2_1 = Momentum4(df['pi2_E_1'], df["pi2_px_1"], df["pi2_py_1"], df["pi2_pz_1"])
            pi3_1 = Momentum4(df['pi3_E_1'], df["pi3_px_1"], df["pi3_py_1"], df["pi3_pz_1"])
            pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
            pi2_2 = Momentum4(df['pi2_E_2'], df["pi2_px_2"], df["pi2_py_2"], df["pi2_pz_2"])
            pi3_2 = Momentum4(df['pi3_E_2'], df["pi3_px_2"], df["pi3_py_2"], df["pi3_pz_2"])
            rest_frame = pi_1 + pi2_1 + pi3_1 + pi_2 + pi2_2 + pi3_2
            boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
            pi_1_boosted = pi_1.boost_particle(boost)
            pi2_1_boosted = pi2_1.boost_particle(boost)
            pi3_1_boosted = pi3_1.boost_particle(boost)
            self.rotationMatrices = self.rotationMatrixVectorised((pi_1_boosted+pi2_1_boosted+pi3_1_boosted)[1:].T, np.tile(np.array([0, 0, 1]), (pi_1_boosted.e.shape[0], 1)))
            return boost
        else:
            raise ValueError('Channel not understood')

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
        "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
        "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
    ]
    DL = DataLoader(variables_rho_rho, 'rho_rho')
    # DL.createRecoData(binary=True, addons=['met'])
    # DL.readGenData()
