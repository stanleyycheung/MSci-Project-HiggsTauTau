# import uproot
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from pylorentz import Momentum4, Position4
from alpha_calculator import AlphaCalculator

class NeutrinoReconstructor:
    """
    Requires: 
    - df_tt_rho_rho.pkl - get from DL.loadRecoData()
    - BR with gen df - constructing now 
    """
    reco_data_dir = './df_tt_rho_rho.pkl'
    gen_data_dir = './df_tt_gen_rho_rho.pkl'
    saved_df_dir = '../stanley/df_saved'
    DEFAULT_VALUE = 0

    def __init__(self, binary, seed=1,):
        np.random.seed(seed)
        self.seed = seed
        self.binary = binary
        self.m_higgs = 125.18
        self.m_tau = 1.776

    def loadRecoData(self, channel='rho_rho', skip=False):
        """
        Loads the .root information of events in the laboratory frame
        """
        if skip:
            return None
        df_tt = pd.read_pickle(NeutrinoReconstructor.reco_data_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['mva_dm_1'] == 1) & (df_tt['mva_dm_2'] == 1) & (
                df_tt["tau_decay_mode_1"] == 1) & (df_tt["tau_decay_mode_2"] == 1)]
        # TODO: To add other channels
        df_reco = df.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2",
                           "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        return df_reco


    def loadBRGenData(self):
        return pd.read_pickle(f'{NeutrinoReconstructor.saved_df_dir}/rho_rho/df_rho_rho.pkl')


    def runAlphaReconstructor(self, df_reco_gen, df_br, load_alpha, termination=1000):
        """
        Calculates the alphas and reconstructs neutrino momenta
        df_reco_gen - events straight from the .root file with gen info
        df_br - BR events
        Default error value: -1
        To do:
        - include azimuthal angles of the neutrinos
        Notes:
        - not inserting four-vector -> other components are just 0
        Returns: alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2
        """
        AC = AlphaCalculator(df_reco_gen, df_br, self.binary, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed, default_value=NeutrinoReconstructor.DEFAULT_VALUE)
        alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = AC.runAlpha(termination=termination)
        # alpha_1, alpha_2 = AC.runAlpha(termination=termination)
        # alpha_1, alpha_2 = AC.runAlphaOld(termination=termination)
        # p_z_nu_1 = alpha_1*(df_br.pi_pz_1_br + df_br.pi0_pz_1_br)
        # p_z_nu_2 = alpha_2*(df_br.pi_pz_2_br + df_br.pi0_pz_2_br)
        # E_nu_1 = (self.m_tau**2 - (df_br.pi_E_1_br+df_br.pi0_E_1_br)**2 + (df_br.pi_pz_1_br + df_br.pi0_pz_1_br)
        #           ** 2 + 2*p_z_nu_1*(df_br.pi_pz_1_br + df_br.pi0_pz_1_br))/(2*(df_br.pi_E_1_br+df_br.pi0_E_1_br))
        # E_nu_2 = (self.m_tau**2 - (df_br.pi_E_2_br+df_br.pi0_E_2_br)**2 + (df_br.pi_pz_2_br + df_br.pi0_pz_2_br)
        #           ** 2 + 2*p_z_nu_2*(df_br.pi_pz_2_br + df_br.pi0_pz_2_br))/(2*(df_br.pi_E_2_br+df_br.pi0_E_2_br))
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        p_t_nu_1[np.isnan(p_t_nu_1)] = NeutrinoReconstructor.DEFAULT_VALUE
        p_t_nu_2[np.isnan(p_t_nu_2)] = NeutrinoReconstructor.DEFAULT_VALUE
        # populate input df with neutrino variables
        # df_br['E_nu_1'] = E_nu_1
        # df_br['E_nu_2'] = E_nu_2
        # df_br['p_t_nu_1'] = p_t_nu_1
        # df_br['p_t_nu_2'] = p_t_nu_2
        # df_br['p_z_nu_1'] = p_z_nu_1
        # df_br['p_z_nu_2'] = p_z_nu_2
        return alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2

    def test1(self, df_reco_gen, df_br, load_alpha, termination=1000):
        """
        Runs test from old criteria
        """
        # df_reco_gen = self.loadRecoData(skip=load_alpha)
        AC = AlphaCalculator(df_reco_gen, df_br, self.binary, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed)
        alpha_1, alpha_2 = AC.runAlphaOld(termination=termination)
        df = df_br
        p_z_nu_1 = alpha_1*(df.pi_pz_1_br + df.pi0_pz_1_br)
        p_z_nu_2 = alpha_2*(df.pi_pz_2_br + df.pi0_pz_2_br)
        E_nu_1 = (self.m_tau**2 - (df.pi_E_1_br+df.pi0_E_1_br)**2 + (df.pi_pz_1_br + df.pi0_pz_1_br)
                  ** 2 + 2*p_z_nu_1*(df.pi_pz_1_br + df.pi0_pz_1_br))/(2*(df.pi_E_1_br+df.pi0_E_1_br))
        E_nu_2 = (self.m_tau**2 - (df.pi_E_2_br+df.pi0_E_2_br)**2 + (df.pi_pz_2_br + df.pi0_pz_2_br)
                  ** 2 + 2*p_z_nu_2*(df.pi_pz_2_br + df.pi0_pz_2_br))/(2*(df.pi_E_2_br+df.pi0_E_2_br))
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        p_t_nu_1[np.isnan(p_t_nu_1)] = -1
        p_t_nu_2[np.isnan(p_t_nu_2)] = -1
        df['alpha_1'] = alpha_1
        df['alpha_2'] = alpha_2
        df['E_nu_1'] = E_nu_1
        df['E_nu_2'] = E_nu_2
        df['p_t_nu_1'] = p_t_nu_1
        df['p_t_nu_2'] = p_t_nu_2
        df['p_z_nu_1'] = p_z_nu_1
        df['p_z_nu_2'] = p_z_nu_2
        pd.to_pickle(df_br, 'misc/df_br_old.pkl')

    def test2(self, termination=10000):
        """
        For testing!
        Calculates the alphas and reconstructs neutrino momenta
        Default error value: -1
        """
        # to include high alpha constraints
        load_alpha = True
        df_reco = self.loadRecoData(skip=load_alpha)
        df = self.loadBRGenData()
        # df and df_reco are nearly the same?
        AC = AlphaCalculator(df_reco, df, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed)
        alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = AC.runAlpha(termination=termination)
        # p_z_nu_1 = alpha_1*(df.pi_pz_1_br + df.pi0_pz_1_br)
        # p_z_nu_2 = alpha_2*(df.pi_pz_2_br + df.pi0_pz_2_br)
        # E_nu_1 = (self.m_tau**2 - (df.pi_E_1_br+df.pi0_E_1_br)**2 + (df.pi_pz_1_br + df.pi0_pz_1_br)
        #           ** 2 + 2*p_z_nu_1*(df.pi_pz_1_br + df.pi0_pz_1_br))/(2*(df.pi_E_1_br+df.pi0_E_1_br))
        # E_nu_2 = (self.m_tau**2 - (df.pi_E_2_br+df.pi0_E_2_br)**2 + (df.pi_pz_2_br + df.pi0_pz_2_br)
        #           ** 2 + 2*p_z_nu_2*(df.pi_pz_2_br + df.pi0_pz_2_br))/(2*(df.pi_E_2_br+df.pi0_E_2_br))
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        p_t_nu_1[np.isnan(p_t_nu_1)] = -1
        p_t_nu_2[np.isnan(p_t_nu_2)] = -1
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
        # AC.checkAlphaPz(df_red)
        # AC.profileAlphaPz(df_red)
        return df, df_red

    def evaluateNegative(self, var):
        print(f"Fraction of < 0: {(var<0).sum()/len(var):.3f}")


if __name__ == '__main__':
    from data_loader import DataLoader
    NR = NeutrinoReconstructor(binary=False)
    # NR.testRunAlphaReconstructor()
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
    channel = 'rho_rho'
    DL = DataLoader(variables_rho_rho, channel)
    df, df_rho_ps, df_rho_sm = DL.cleanRecoData(DL.readRecoData(from_pickle=True))
    df_br = DL.loadRecoData(binary=False).reset_index(drop=True)
    # augment the binary df
    df_reco_gen, _ = DL.augmentDfToBinary(df_rho_ps, df_rho_sm)
    # slightly different lengths - due to binary/non_binary

    # debug = pd.read_pickle('./misc/debugging_2.pkl')

    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(debug, debug, load_alpha=False, termination=100)
    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(df_reco_gen, df_br, load_alpha=False, termination=100)
    alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=False, termination=100)
    df_br['alpha_1'] = alpha_1
    df_br['alpha_2'] = alpha_2
    df_br['E_nu_1'] = E_nu_1
    df_br['E_nu_2'] = E_nu_2
    df_br['p_t_nu_1'] = p_t_nu_1
    df_br['p_t_nu_2'] = p_t_nu_2
    df_br['p_z_nu_1'] = p_z_nu_1
    df_br['p_z_nu_2'] = p_z_nu_2
    # print(df_br.columns)
    pd.to_pickle(df_br, 'misc/df_br.pkl')
    # NR.test1(df.reset_index(drop=False), df_br, load_alpha=False, termination=1000)


    # THIS COMBINATION WORKS -> DON'T KNOW WHY


