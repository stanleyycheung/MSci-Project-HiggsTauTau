import uproot
import pandas as pd
import numpy as np
from pylorentz import Momentum4
import os


class DataLoader:
    """
    DataLoader class
    Functions:
    - Reads .root file (and saves df if necessary)
    - Selects appropirate channel, and distinguishes sm/ps events
    - Boosts, rotates events into correct rest frame, according to channel
    - Outputs df with all of the possible NN inputs

    To do:
    - Add in functions to read gen data
    - Add in neutrino data
    """

    def __init__(self, variables, channel, input_df_save_dir='./input_df'):
        self.channel = channel
        self.variables = variables
        self.reco_root_path = "./MVAFILE_AllHiggs_tt_new.root"
        if os.path.exists("C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"):
            print('using Kristof\'s .root file')
            self.reco_root_path = "C:\\Users\\krist\\Downloads\\MVAFILE_ALLHiggs_tt_new.root"
        self.reco_df_path = './df_tt'
        self.input_df_save_dir = input_df_save_dir

    def loadRecoData(self, binary):
        print('Reading df pkl file')
        pickle_file_name = f'{self.input_df_save_dir}/input_{self.channel}'
        if binary:
            pickle_file_name += '_b'
        df_inputs = pd.read_pickle(pickle_file_name+'.pkl')
        return df_inputs

    def createRecoData(self, binary,  from_pickle=True, addons=[]):
        print('Loading .root info')
        df = self.readRecoData(from_pickle=from_pickle)
        print('Cleaning data')
        df_clean, df_ps_clean, df_sm_clean = self.cleanRecoData(df)
        print('Creating input data')
        print(df_ps_clean.to_numpy().shape)
        print(df_sm_clean.to_numpy().shape)
        df_inputs = self.createTrainTestData(df_clean, df_ps_clean, df_sm_clean, binary, addons, save=True)
        return df_inputs

    def readRecoData(self, from_pickle=False):
        if not from_pickle:
            tree_tt = uproot.open(self.reco_root_path)["ntuple"]
            # tree_et = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_et.root")["ntuple"]
            # tree_mt = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_mt.root")["ntuple"]
            df = tree_tt.pandas.df(self.variables)
            df.to_pickle(f"{self.reco_df_path}_{self.channel}.pkl")
        else:
            df = pd.read_pickle(f"{self.reco_df_path}_{self.channel}.pkl")
        return df

    def cleanRecoData(self, df):
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

    def createTrainTestData(self, df, df_ps, df_sm, binary, addons=[], save=True):
        if binary:
            print('In binary mode')
            y_sm = pd.DataFrame(np.ones(df_sm.shape[0]))
            y_ps = pd.DataFrame(np.zeros(df_ps.shape[0]))
            y = pd.concat([y_sm, y_ps]).to_numpy()
            df = pd.concat([df_sm, df_ps])
        if self.channel == 'rho_rho':
            df_inputs_data, boost = self.calculateRhoRhoData(df)
        elif self.channel == 'rho_a1':
            df_inputs_data, boost = self.calculateRhoA1Data(df)
        else:
            # no need to check here as checked in cleanRecoData
            df_inputs_data, boost = self.calculateA1A1Data(df)
        df_inputs = pd.DataFrame(df_inputs_data)
        if binary:
            df_inputs['y'] = y
        if not addons:
            self.createAddons(addons, df, df_inputs, boost=boost)
        if save:
            print('Saving df to pickle')
            pickle_file_name = f'{self.input_df_save_dir}/input_{self.channel}'
            if binary:
                pickle_file_name += '_b'
            df_inputs.to_pickle(pickle_file_name+'.pkl')
        return df_inputs

    def calculateRhoRhoData(self, df):
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

    def calculateRhoA1Data(self, df):
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
        #rest_frame = pi_1 + pi_2 + pi0_1 + pi2_2 + pi3_2
        rest_frame = pi_1 + pi_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        pi_1_boosted = pi_1.boost_particle(boost)
        pi_2_boosted = pi_2.boost_particle(boost)
        pi0_1_boosted = pi0_1.boost_particle(boost)
        pi2_2_boosted = pi2_2.boost_particle(boost)
        pi3_2_boosted = pi3_2.boost_particle(boost)
        rho_1_boosted = pi_1_boosted + pi0_1_boosted
        rho_2_boosted = pi_2_boosted + pi3_2_boosted
        a1_boosted = rho_2_boosted + pi2_2_boosted
        # rotations
        # !!! code missing here
        df_inputs_data = {}
        boost = None
        return df_inputs_data, boost

    def calculateA1A1Data(self, df):
        # TODO: include this channel
        df_inputs_data = {}
        boost = None
        return df_inputs_data, boost

    def createAddons(self, addons, df, df_inputs, **kwargs):
        """
        If you want to create more addon features, put the necessary arguments through kwargs, 
        unpack them at the start of this function, and add an if case to your needs
        """
        boost = kwargs["boost"]
        for addon in addons:
            if addon == 'met':
                print('Addon MET loaded')
                E_miss, E_miss_x, E_miss_y = self.addonMET(df, boost)
                df_inputs['E_miss'] = E_miss
                df_inputs['E_miss_x'] = E_miss_x
                df_inputs['E_miss_y'] = E_miss_y

    def addonMET(self, df, boost):
        N = len(df['metx'])
        met_x = Momentum4(df['metx'], np.zeros(N), np.zeros(N), np.zeros(N))
        met_y = Momentum4(df['mety'], np.zeros(N), np.zeros(N), np.zeros(N))
        met = Momentum4(df['met'], np.zeros(N), np.zeros(N), np.zeros(N))
        # boost MET - E_miss is already boosted into the hadronic rest frame
        E_miss = met_x.boost_particle(boost)[0]
        E_miss_x = met_y.boost_particle(boost)[0]
        E_miss_y = met.boost_particle(boost)[0]
        return E_miss, E_miss_x, E_miss_y

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
    variables = [
        "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
        "aco_angle_1", "aco_angle_5", "aco_angle_6", "aco_angle_7",
        "mva_dm_1", "mva_dm_2",
        "tau_decay_mode_1", "tau_decay_mode_2",
        "ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2",  # ignore impact parameter for now
        "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
        "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
        "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
        "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
        "y_1_1", "y_1_2",
        'met', 'metx', 'mety',
        'metcov00', 'metcov01', 'metcov10', 'metcov11',
        #             'sv_x_1', 'sv_y_1', 'sv_z_1', 'sv_x_2', 'sv_y_2','sv_z_2'
    ]
    DL = DataLoader(variables, 'rho_rho')
    DL.createRecoData(binary=True, addons=['met'])
