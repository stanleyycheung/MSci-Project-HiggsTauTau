import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
import matplotlib.pyplot as plt
from utils import profileplot, sps, profileplot_plain
import config
from tqdm import tqdm

class AlphaCalculator:
    # changed file paths to class variables
    pickle_dir = './df_tt_rho_rho.pkl'
    alpha_save_dir = './alpha_analysis'

    def __init__(self, channel, df, df_br, binary, m_higgs, m_tau, default_value, load=False, seed=config.seed_value):
        np.random.seed(seed)
        self.channel = channel
        self.m_higgs = m_higgs
        self.m_tau = m_tau
        self.binary = binary
        self.df = df
        self.df_br = df_br
        self.load = load
        self.DEFAULT_VALUE = default_value

    def loadData(self, channel='rho_rho'):
        df_tt = pd.read_pickle(AlphaCalculator.pickle_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['mva_dm_1'] == 1) & (df_tt['mva_dm_2'] == 1) & (
                df_tt["tau_decay_mode_1"] == 1) & (df_tt["tau_decay_mode_2"] == 1)]
        self.df = df.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2",
                           "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def runAlpha(self, termination=1000):
        """
        Runs alpha calculation, and automatically saves them
        -- Return type: alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 --
        -- Changed to not return neutrino reconstructed info (15/01) --
        -- Returns: alpha_1, alpha_2 --
        Changed back return type to original (19/01)
        Return type: alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
        """
        print('Running alphas')
        if self.load:
            binary_str = ''
            if self.binary:
                binary_str += "_b"
            self.alpha_1 = np.load(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_reco_{termination}'+binary_str+".npy", allow_pickle=True)
            self.alpha_2 = np.load(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_2_reco_{termination}'+binary_str+".npy", allow_pickle=True)
            idx = self.alpha_1==self.DEFAULT_VALUE
            if self.channel == 'rho_rho':
                # deal with BR data
                had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
                had_2_br = self.df_br.rho_E_2_br.to_numpy(), self.df_br.rho_pz_2_br.to_numpy()
            elif self.channel == 'rho_a1':
                had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
                had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
            elif self.channel == 'a1_a1':
                had_1_br = self.df_br.a1_E_1_br.to_numpy(), self.df_br.a1_pz_1_br.to_numpy()
                had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
            else:
                raise ValueError('Channel not understood')
            had_E_1_br, had_pz_1_br = had_1_br
            had_E_2_br, had_pz_2_br = had_2_br
            p_z_nu_1 = self.alpha_1*(had_pz_1_br)
            p_z_nu_2 = self.alpha_2*(had_pz_2_br)
            E_nu_1 = (self.m_tau**2 - (had_E_1_br)**2 + (had_pz_1_br)** 2 + 2*p_z_nu_1*(had_pz_1_br))/(2*(had_E_1_br))
            E_nu_2 = (self.m_tau**2 - (had_E_2_br)**2 + (had_pz_2_br)** 2 + 2*p_z_nu_2*(had_pz_2_br))/(2*(had_E_2_br))
            p_z_nu_1[idx] = self.DEFAULT_VALUE
            p_z_nu_2[idx] = self.DEFAULT_VALUE
            E_nu_1[idx] = self.DEFAULT_VALUE
            E_nu_2[idx] = self.DEFAULT_VALUE
            return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
            # return self.alpha_1, self.alpha_2
        self.alpha_1, self.alpha_2 = [], []
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = [], [], [], []
        rejection = 0
        E_miss_x = self.df.metx.to_numpy()
        E_miss_y = self.df.mety.to_numpy()
        metcov00 = self.df.metcov00.to_numpy()
        metcov01 = self.df.metcov01.to_numpy()
        metcov10 = self.df.metcov10.to_numpy()
        metcov11 = self.df.metcov11.to_numpy()
        if self.channel == 'rho_rho':
            # deal with lab frame data
            had_E_1 = (self.df.pi_E_1 + self.df.pi0_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi0_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi0_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi0_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi0_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi0_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi0_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi0_pz_2).to_numpy()
            # deal with BR data
            had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
            had_2_br = self.df_br.rho_E_2_br.to_numpy(), self.df_br.rho_pz_2_br.to_numpy()
        elif self.channel == 'rho_a1':
            # deal with lab frame data
            had_E_1 = (self.df.pi_E_1 + self.df.pi0_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi0_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi0_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi0_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi2_E_2 + self.df.pi3_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi2_px_2 + self.df.pi3_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi2_py_2 + self.df.pi3_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi2_pz_2 + self.df.pi3_pz_2).to_numpy()
            # deal with BR data
            had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
            had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
        elif self.channel == 'a1_a1':
            had_E_1 = (self.df.pi_E_1 + self.df.pi2_E_1 + self.df.pi3_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi2_px_1 + self.df.pi3_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi2_py_1 + self.df.pi3_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi2_pz_1 + self.df.pi3_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi2_E_2 + self.df.pi3_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi2_px_2 + self.df.pi3_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi2_py_2 + self.df.pi3_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi2_pz_2 + self.df.pi3_pz_2).to_numpy()
            had_1_br = self.df_br.a1_E_1_br.to_numpy(), self.df_br.a1_pz_1_br.to_numpy()
            had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
        else:
            raise ValueError('Channel not understood')

        for i in tqdm(range(self.df.shape[0])):
            had_1_row = np.array([had_E_1[i], had_px_1[i], had_py_1[i], had_pz_1[i]])
            had_2_row = np.array([had_E_2[i], had_px_2[i], had_py_2[i], had_pz_2[i]])
            row_cov = np.array(([metcov00[i], metcov01[i]], [metcov10[i], metcov11[i]]))
            (alpha_1_loc, alpha_2_loc), (p_z_nu_1_loc, E_nu_1_loc, p_z_nu_2_loc, E_nu_2_loc) = self.getAlpha(i, E_miss_x[i], E_miss_y[i], had_1_row, had_2_row, had_1_br, had_2_br, row_cov, termination=termination)
            p_z_nu_1.append(p_z_nu_1_loc)
            E_nu_1.append(E_nu_1_loc)
            p_z_nu_2.append(p_z_nu_2_loc)
            E_nu_2.append(E_nu_2_loc)
            self.alpha_1.append(alpha_1_loc)
            self.alpha_2.append(alpha_2_loc)
            if alpha_1_loc == self.DEFAULT_VALUE:
                rejection += 1
            # if i%10000 == 0:
            #     print(f'getting alpha for {i}, rejection: {rejection}/{self.df.shape[0]}')
        print(f'Total rejection: {rejection}/{self.df.shape[0]}')
        print('Saving alpha')
        binary_str = ''
        if self.binary:
            binary_str += "_b"
        np.save(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_reco_{termination}'+binary_str+".npy", self.alpha_1, allow_pickle=True)
        np.save(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_2_reco_{termination}'+binary_str+".npy", self.alpha_2, allow_pickle=True)
        return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2

    def runAlphaGen(self):
        print('Running alphas')
        if self.load:
            binary_str = ''
            if self.binary:
                binary_str += "_b"
            self.alpha_1 = np.load(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_gen'+binary_str+".npy", allow_pickle=True)
            self.alpha_2 = np.load(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_gen'+binary_str+".npy", allow_pickle=True)
            idx = self.alpha_1==self.DEFAULT_VALUE
            if self.channel == 'rho_rho':
                # deal with BR data
                had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
                had_2_br = self.df_br.rho_E_2_br.to_numpy(), self.df_br.rho_pz_2_br.to_numpy()
            elif self.channel == 'rho_a1':
                had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
                had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
            elif self.channel == 'a1_a1':
                had_1_br = self.df_br.a1_E_1_br.to_numpy(), self.df_br.a1_pz_1_br.to_numpy()
                had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
            else:
                raise ValueError('Channel not understood')
            had_E_1_br, had_pz_1_br = had_1_br
            had_E_2_br, had_pz_2_br = had_2_br
            p_z_nu_1 = self.alpha_1*(had_pz_1_br)
            p_z_nu_2 = self.alpha_2*(had_pz_2_br)
            E_nu_1 = (self.m_tau**2 - (had_E_1_br)**2 + (had_pz_1_br)** 2 + 2*p_z_nu_1*(had_pz_1_br))/(2*(had_E_1_br))
            E_nu_2 = (self.m_tau**2 - (had_E_2_br)**2 + (had_pz_2_br)** 2 + 2*p_z_nu_2*(had_pz_2_br))/(2*(had_E_2_br))
            p_z_nu_1[idx] = self.DEFAULT_VALUE
            p_z_nu_2[idx] = self.DEFAULT_VALUE
            E_nu_1[idx] = self.DEFAULT_VALUE
            E_nu_2[idx] = self.DEFAULT_VALUE
            return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
        self.alpha_1, self.alpha_2 = [], []
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = [], [], [], []
        rejection = 0
        E_miss_x = self.df.metx.to_numpy()
        E_miss_y = self.df.mety.to_numpy()
        if self.channel == 'rho_rho':
            # deal with lab frame data
            had_E_1 = (self.df.pi_E_1 + self.df.pi0_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi0_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi0_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi0_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi0_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi0_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi0_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi0_pz_2).to_numpy()
            # deal with BR data
            had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
            had_2_br = self.df_br.rho_E_2_br.to_numpy(), self.df_br.rho_pz_2_br.to_numpy()
        elif self.channel == 'rho_a1':
            # deal with lab frame data
            had_E_1 = (self.df.pi_E_1 + self.df.pi0_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi0_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi0_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi0_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi2_E_2 + self.df.pi3_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi2_px_2 + self.df.pi3_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi2_py_2 + self.df.pi3_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi2_pz_2 + self.df.pi3_pz_2).to_numpy()
            # deal with BR data
            had_1_br = self.df_br.rho_E_1_br.to_numpy(), self.df_br.rho_pz_1_br.to_numpy()
            had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
        elif self.channel == 'a1_a1':
            had_E_1 = (self.df.pi_E_1 + self.df.pi2_E_1 + self.df.pi3_E_1).to_numpy()
            had_px_1 = (self.df.pi_px_1 + self.df.pi2_px_1 + self.df.pi3_px_1).to_numpy()
            had_py_1 = (self.df.pi_py_1 + self.df.pi2_py_1 + self.df.pi3_py_1).to_numpy()
            had_pz_1 = (self.df.pi_pz_1 + self.df.pi2_pz_1 + self.df.pi3_pz_1).to_numpy()
            had_E_2 = (self.df.pi_E_2 + self.df.pi2_E_2 + self.df.pi3_E_2).to_numpy()
            had_px_2 = (self.df.pi_px_2 + self.df.pi2_px_2 + self.df.pi3_px_2).to_numpy()
            had_py_2 = (self.df.pi_py_2 + self.df.pi2_py_2 + self.df.pi3_py_2).to_numpy()
            had_pz_2 = (self.df.pi_pz_2 + self.df.pi2_pz_2 + self.df.pi3_pz_2).to_numpy()
            had_1_br = self.df_br.a1_E_1_br.to_numpy(), self.df_br.a1_pz_1_br.to_numpy()
            had_2_br = self.df_br.a1_E_2_br.to_numpy(), self.df_br.a1_pz_2_br.to_numpy()
        else:
            raise ValueError('Channel not understood')
        for i in tqdm(range(self.df.shape[0])):
            had_1_row = np.array([had_E_1[i], had_px_1[i], had_py_1[i], had_pz_1[i]])
            had_2_row = np.array([had_E_2[i], had_px_2[i], had_py_2[i], had_pz_2[i]])
            (alpha_1_loc, alpha_2_loc), (p_z_nu_1_loc, E_nu_1_loc, p_z_nu_2_loc, E_nu_2_loc) = self.getAlphaGen(i, E_miss_x[i], E_miss_y[i], had_1_row, had_2_row, had_1_br, had_2_br)
            p_z_nu_1.append(p_z_nu_1_loc)
            E_nu_1.append(E_nu_1_loc)
            p_z_nu_2.append(p_z_nu_2_loc)
            E_nu_2.append(E_nu_2_loc)
            self.alpha_1.append(alpha_1_loc)
            self.alpha_2.append(alpha_2_loc)
            if alpha_1_loc == self.DEFAULT_VALUE:
                rejection += 1
            # if i%10000 == 0:
            #     print(f'getting alpha for {i}, rejection: {rejection}/{self.df.shape[0]}')
        print(f'Total rejection: {rejection}/{self.df.shape[0]}')
        print('Saving alpha')
        binary_str = ''
        if self.binary:
            binary_str += "_b"
        np.save(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_gen'+binary_str+".npy", self.alpha_1, allow_pickle=True)
        np.save(f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_2_gen'+binary_str+".npy", self.alpha_2, allow_pickle=True)
        return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2


    def getAlpha(self, idx, E_miss_x, E_miss_y, had_1_row, had_2_row, had_1_br, had_2_br, cov, mode=1, termination=1000):
        """
        Calculates alpha with constraints, returns self.DEFAULT_VALUE if not possible
        Returns: (alpha_1, alpha_2), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
        """
        had_E_1_br, had_pz_1_br = had_1_br
        had_E_2_br, had_pz_2_br = had_2_br
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, had_1_row, had_2_row, mode)
        # p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1, alpha_2)
        mean = np.array([E_miss_x, E_miss_y])
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1, alpha_2, had_pz_1_br, had_pz_2_br, had_E_1_br, had_E_2_br)
        if alpha_1 < 0 or alpha_2 < 0 or np.abs(E_nu_1) < np.abs(p_z_nu_1) or np.abs(E_nu_2) < np.abs(p_z_nu_2) or E_nu_1 < 0 or E_nu_2 < 0:
        # if alpha_1 < 0 or alpha_2 < 0:
        # if alpha_1 < 0 or alpha_2 < 0 or np.abs(E_nu_1) < np.abs(p_z_nu_1) or np.abs(E_nu_2) < np.abs(p_z_nu_2):
            E_miss_gen = np.random.multivariate_normal(mean, cov, termination)
        else:
            return (alpha_1, alpha_2), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
            # return alpha_1, alpha_2
        # turn into vectorised calculations
        E_miss_x = E_miss_gen[:, 0]
        E_miss_y = E_miss_gen[:, 1]
        # alpha is now an array shape = (termination)
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, had_1_row, had_2_row, mode)
        for i in range(termination):
            E_miss_x, E_miss_y = E_miss_gen[i]
            # print(E_miss_x, E_miss_y, alpha_1, alpha_2)
            # p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1[i], alpha_2[i])
            p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1[i], alpha_2[i], had_pz_1_br, had_pz_2_br, had_E_1_br, had_E_2_br)
            if alpha_1[i] > 0 and alpha_2[i] > 0 and np.abs(E_nu_1) > np.abs(p_z_nu_1) and np.abs(E_nu_2) > np.abs(p_z_nu_2) and E_nu_1 > 0 and E_nu_2 > 0:
            # if alpha_1[i] > 0 and alpha_2[i] > 0:
            # if alpha_1[i] > 0 and alpha_2[i] > 0 and np.abs(E_nu_1) > np.abs(p_z_nu_1) and np.abs(E_nu_2) > np.abs(p_z_nu_2):
                # return (alpha_1[i], alpha_2[i]), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
                return (alpha_1[i], alpha_2[i]), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
        # return (-1, -1), (-1, -1, -1, -1)
        # return self.DEFAULT_VALUE, self.DEFAULT_VALUE
        return (self.DEFAULT_VALUE, self.DEFAULT_VALUE), (self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE)

    def getAlphaGen(self, idx, E_miss_x, E_miss_y, had_1_row, had_2_row, had_1_br, had_2_br, mode=1):
        had_E_1_br, had_pz_1_br = had_1_br
        had_E_2_br, had_pz_2_br = had_2_br
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, had_1_row, had_2_row, mode)
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1, alpha_2, had_pz_1_br, had_pz_2_br, had_E_1_br, had_E_2_br)
        if alpha_1 < 0 or alpha_2 < 0 or np.abs(E_nu_1) < np.abs(p_z_nu_1) or np.abs(E_nu_2) < np.abs(p_z_nu_2) or E_nu_1 < 0 or E_nu_2 < 0:
            return (self.DEFAULT_VALUE, self.DEFAULT_VALUE), (self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE)
        else:
            return (alpha_1, alpha_2), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)

    def getReconstructedInfo(self, i, alpha_1, alpha_2, had_pz_1_br, had_pz_2_br, had_E_1_br, had_E_2_br):
        """
        Reconstructs the momenta of neutrinos in the BR frame
        """
        p_z_nu_1 = alpha_1*(had_pz_1_br[i])
        p_z_nu_2 = alpha_2*(had_pz_2_br[i])
        E_nu_1 = (self.m_tau**2 - (had_E_1_br[i])**2 + (had_pz_1_br[i])** 2 + 2*p_z_nu_1*(had_pz_1_br[i]))/(2*(had_E_1_br[i]))
        E_nu_2 = (self.m_tau**2 - (had_E_2_br[i])**2 + (had_pz_2_br[i])** 2 + 2*p_z_nu_2*(had_pz_2_br[i]))/(2*(had_E_2_br[i]))
        return p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2

    def profileAlphaPz(self, df_red, termination=100):
        # TODO: CHANGE DUE TO CHANNEL
        # remove NaN values
        df_red.dropna(inplace=True)
        p_z_nu_1 = df_red.p_z_nu_1.to_numpy()
        # note that br has NaN values that correspond to 9999 values
        gen_p_z_nu_1 = df_red.gen_nu_pz_1_br.to_numpy()
        error = (gen_p_z_nu_1-p_z_nu_1)       
        rel_error = (gen_p_z_nu_1-p_z_nu_1)/gen_p_z_nu_1
        profileplot(p_z_nu_1, error, bins=500, xlabel='reconstructed p_z', ylabel='error')
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/err_profile_{termination}.png')
        profileplot(p_z_nu_1, rel_error, bins=500, xlabel='reconstructed p_z', ylabel='rel error', mode=0)
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/relerr_profile_{termination}.png')
        # plt.hist(gen_plot)
        # plt.hist(reco_plot_1)
        plt.show()

    def checkAlphaPz(self, df_red, termination=100):
        # TODO: CHANGE DUE TO CHANNEL
        # remove NaN values
        df_red.dropna(inplace=True)
        p_z_nu_1 = df_red.p_z_nu_1.to_numpy()
        # alpha_1 = df_red.alpha_1.to_numpy()
        # p_z_nu_2 = df_red.p_z_nu_2.to_numpy()
        gen_p_z_nu_1 = df_red.gen_nu_pz_1_br.to_numpy()
        # remove NaN values
        fig1, ax1 = plt.subplots(figsize=(8,6))
        bins = np.linspace(0, 1000, 500)
        plt.hist(p_z_nu_1, alpha=0.5, bins=bins, label='from alpha')
        plt.hist(gen_p_z_nu_1, alpha=0.5, bins=bins, label='from gen')
        ax1.legend()
        ax1.set_ylabel('Freq')
        ax1.set_xlabel('p_z_nu_1')
        fig2, ax2 = plt.subplots(figsize=(8,6))
        rel_error = (gen_p_z_nu_1-p_z_nu_1)/gen_p_z_nu_1
        upper_lim, lower_lim = 2, -2
        # rel_error_red = rel_error[np.where((rel_error >= lower_lim) & (rel_error <= upper_lim))]
        ax2.hist(rel_error, bins=100, label=f'termination={termination}')
        # ax2.hist(np.clip(rel_error, -5, 5), bins=100, label=f'termination={termination}')
        ax2.set_ylabel('Freq')
        ax2.set_xlabel('Relative error')
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/relerr_{termination}.png')
        fig3, ax3 = plt.subplots(figsize=(8,6))
        error = (gen_p_z_nu_1-p_z_nu_1)
        ax3.hist(np.clip(error, -200, 200), bins=100)
        ax3.set_ylabel('Freq')
        ax3.set_xlabel('Error')
        plt.show()
        """
        fig1, ax1 = plt.subplots(figsize=(8,6))
        ax1.hist(reco_plot_1[np.where((reco_plot_1 >= -10) & (reco_plot_1 <= 500))])
        ax1.hist(gen_plot[np.where((gen_plot >= -10) & (gen_plot <= 500))], alpha=0.5, bins=100, label='from gen')
        ax1.legend()
        ax1.set_ylabel('Freq')
        ax1.set_xlabel('p_z_nu_1')
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/freq_{termination}.png')
        fig2, ax2 = plt.subplots(figsize=(8,6))
        rel_error = (gen_plot-reco_plot_1)/gen_plot
        # plt.hist(np.clip(rel_error, -1, 1), bins=500, label='termination=1000')
        upper_lim, lower_lim = 2, -2
        rel_error_red = rel_error[np.where((rel_error >= lower_lim) & (rel_error <= upper_lim))]
        ax2.hist(rel_error_red, bins=100, label=f'termination={termination}')
        ax2.set_ylabel('Freq')
        ax2.set_xlabel('Relative error')
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/relerr_{termination}.png')
        fig3, ax3 = plt.subplots(figsize=(8,6))
        error = (gen_plot-reco_plot_1)
        ax3.hist(np.clip(error, -200, 200), bins=100)
        ax3.set_ylabel('Freq')
        ax3.set_xlabel('Error')
        plt.savefig(f'{AlphaCalculator.alpha_save_dir}/err_{termination}.png')
        plt.tight_layout()
        plt.show()
        """

    def calcAlpha(self, E_miss_x, E_miss_y, had_1, had_2, mode):
        # rhos are 4 vectors, not 3 vectors
        if mode == 1:
            alpha_2 = (E_miss_y*had_1[1]-E_miss_x*had_1[2])/(had_2[2]*had_1[1]-had_2[1]*had_1[2])
            alpha_1 = (E_miss_x - alpha_2*had_2[1])/had_1[1]
        elif mode == 2:
            alpha_1 = (E_miss_y*had_2[1]-E_miss_x*had_2[2])/(had_1[2]*had_2[1]-had_1[1]*had_2[2])
            alpha_2 = (self.m_higgs**2/2 - self.m_tau**2)/(had_1[0]*had_2[0]-had_1[1]*had_2[1]-had_1[2]*had_1[2]-had_1[3]*had_1[3])/(1+alpha_1) - 1
        elif mode == 3:
            alpha_2 = (E_miss_y*had_1[1]-E_miss_x*had_1[2])/(had_2[2]*had_1[1]-had_2[1]*had_1[2])
            alpha_1 = (self.m_higgs**2/2 - self.m_tau**2)/(had_1[0]*had_2[0]-had_1[1]*had_2[1]-had_1[2]*had_1[2]-had_1[3]*had_1[3])/(1+alpha_2) - 1
        else:
            raise ValueError('incorrect mode in parameters')
        return alpha_1, alpha_2 

