# import uproot
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from pylorentz import Momentum4, Position4
from alpha_calculator import AlphaCalculator


class NeutrinoReconstructor:
    def __init__(self, seed=1):
        np.random.seed(seed)
        self.seed = seed
        self.reco_data_dir = './df_tt.pkl'
        self.gen_data_dir = './df_tt_gen.pkl'
        self.saved_df_dir = './df_saved'
        self.m_higgs = 125.18
        self.m_tau = 1.776

    def loadRecoData(self, channel='rho_rho', skip=False):
        if skip:
            return None
        df_tt = pd.read_pickle(self.reco_data_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['mva_dm_1'] == 1) & (df_tt['mva_dm_2'] == 1) & (
                df_tt["tau_decay_mode_1"] == 1) & (df_tt["tau_decay_mode_2"] == 1)]
        # TODO: To add other channels
        df_reco = df.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2",
                           "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        return df_reco

    def loadGenData(self, channel='rho_rho', skip=False):
        if skip:
            return None
        df_tt = pd.read_pickle(self.gen_data_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['dm_1'] == 1) & (df_tt['dm_2'] == 1)]
        # TODO: To add other channels
        self.df = df.drop(["dm_1", "dm_2", "wt_cp_sm", "wt_cp_ps",
                           "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def loadBRData(self):
        return pd.read_pickle(f'{self.saved_df_dir}/rho_rho/df_rho_rho.pkl')

    def runAlphaReconstructor(self, termination=10000):
        load_alpha = True
        df_reco = self.loadRecoData(skip=load_alpha)
        df = self.loadBRData()
        AC = AlphaCalculator(df_reco, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed)
        alpha_1, alpha_2 = AC.runAlpha(termination=termination)
        # print(alpha_1, alpha_2)
        p_z_nu_1 = alpha_1*(df.pi_pz_1_br + df.pi0_pz_1_br)
        p_z_nu_2 = alpha_2*df.pi_pz_2_br + df.pi0_pz_2_br
        E_nu_1 = (self.m_tau**2 - (df.pi_E_1_br+df.pi0_E_1_br)**2 + (df.pi_pz_1_br + df.pi0_pz_1_br)
                  ** 2 + 2*p_z_nu_1*(df.pi_pz_1_br + df.pi0_pz_1_br))/(2*(df.pi_E_1_br+df.pi0_E_1_br))
        E_nu_2 = (self.m_tau**2 - (df.pi_E_2_br+df.pi0_E_2_br)**2 + (df.pi_pz_2_br + df.pi0_pz_2_br)
                  ** 2 + 2*p_z_nu_2*(df.pi_pz_2_br + df.pi0_pz_2_br))/(2*(df.pi_E_2_br+df.pi0_E_2_br))
        p_t_nu_1 = np.sqrt(E_nu_1**2 - p_z_nu_1**2)
        p_t_nu_2 = np.sqrt(E_nu_2**2 - p_z_nu_2**2)
        # p_t_nu_1[np.isnan(p_t_nu_1)] = 0
        # p_t_nu_2[np.isnan(p_t_nu_2)] = 0
        # print(self.evaluateNegative(alpha_1), self.evaluateNegative(alpha_2))
        # print(self.evaluateNegative(E_nu_1), self.evaluateNegative(E_nu_2))
        df['alpha_1'] = alpha_1
        df['alpha_2'] = alpha_2
        df['E_nu_1'] = E_nu_1
        df['E_nu_2'] = E_nu_2
        df['p_t_nu_1'] = p_t_nu_1
        df['p_t_nu_2'] = p_t_nu_2
        df['p_z_nu_1'] = p_z_nu_1
        df['p_z_nu_2'] = p_z_nu_2
        # print(p_z_nu_1)
        df_red = df[(df['alpha_1'] > 0) & (df['alpha_2'] > 0) & (
            df['E_nu_1'] > 0) & (df['E_nu_2'] > 0)].reset_index(drop=True)
        df_red.dropna(inplace=True)
        df_red.name = f'{termination}'
        print(f'Rejected {(len(df.index)-len(df_red.index))/len(df_red.index)*100:.4f}% events due to physical reasons')
        AC.checkAlphaPz(df_red, df)
        return df, df_red

    def plotDistribution(self):
        pass

    def evaluateNegative(self, var):
        print(f"Fraction of < 0: {(var<0).sum()/len(var):.3f}")


if __name__ == '__main__':
    NR = NeutrinoReconstructor()
    NR.runAlphaReconstructor()
