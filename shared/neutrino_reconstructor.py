# import uproot
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from pylorentz import Momentum4, Position4
from alpha_calculator import AlphaCalculator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
import config


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

    def __init__(self, binary, seed=config.seed_value):
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

    def boostAndRotateNeutrinos(self, df):
        """
        Returns the neutrino data rotated and boosted
        """
        nu_1 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_1"])), df["gen_nu_eta_1"], df["gen_nu_phi_1"], df["gen_nu_p_1"])
        nu_2 = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_2"])), df["gen_nu_eta_2"], df["gen_nu_phi_2"], df["gen_nu_p_2"])
        pi_1 = Momentum4(df['pi_E_1'], df["pi_px_1"], df["pi_py_1"], df["pi_pz_1"])
        pi_2 = Momentum4(df['pi_E_2'], df["pi_px_2"], df["pi_py_2"], df["pi_pz_2"])
        pi0_1 = Momentum4(df['pi0_E_1'], df["pi0_px_1"], df["pi0_py_1"], df["pi0_pz_1"])
        pi0_2 = Momentum4(df['pi0_E_2'], df["pi0_px_2"], df["pi0_py_2"], df["pi0_pz_2"])
        # boost into rest frame of resonances
        rest_frame = pi_1 + pi_2 + pi0_1 + pi0_2
        boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
        nu_1_boosted = nu_1.boost_particle(boost)
        nu_2_boosted = nu_2.boost_particle(boost)
        rho_1_boosted = pi_1.boost_particle(boost) + pi0_1.boost_particle(boost)
        nu_1_boosted_rot, nu_2_boosted_rot = []
        rotationMatrices = self.rotationMatrixVectorised(rho_1_boosted[1:].T, np.tile(np.array([0, 0, 1]), (rho_1_boosted.e.shape[0], 1)))
        nu_1_boosted_rot = np.einsum('ij,ikj->ik', nu_1_boosted[1:].T, rotationMatrices)
        nu_2_boosted_rot = np.einsum('ij,ikj->ik', nu_2_boosted[1:].T, rotationMatrices)
        return nu_1_boosted_rot, nu_2_boosted_rot

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

    def dealWithMissingData(self, df_br, mode):
        """
        Deals with rejected events according to mode
        Mode:
        0 - Simple flag
        1 - Linear interpolation - BayesianRidge algorithm
        2 - KNN algorithm
        3 - Replace with mean
        """
        print('Imputing missing data')
        # change to return df_br and modify in place
        # alpha_flag = NeutrinoReconstructor.DEFAULT_VALUE + 1
        if mode == 0:
            df_br['flag'] = np.where(df_br['alpha_1']==NeutrinoReconstructor.DEFAULT_VALUE, 0, 1)
            return df_br
        elif mode == 1:
            # df_br['alpha_1'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
            # df_br['alpha_2'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
            # print(df_br.head())
            # default is BayesianRidge
            # itImp = IterativeImputer(missing_values=alpha_flag, random_state=0, verbose=1)
            itImp = IterativeImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE, random_state=0, verbose=1)
            df_br_imputed = pd.DataFrame(itImp.fit_transform(df_br), columns=df_br.columns)
            return df_br_imputed
            # return self.calculateFromAlpha(df_br_imputed, df_br_imputed['alpha_1'], df_br_imputed['alpha_2'])
        elif mode == 2:
            # df_br['alpha_1'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
            # df_br['alpha_2'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
            # KNNImp = KNNImputer(missing_values=alpha_flag, n_neighbors=2) 
            KNNImp = KNNImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE, n_neighbors=2)
            df_br_imputed = pd.DataFrame(KNNImp.fit_transform(df_br), columns=df_br.columns)
            return df_br_imputed
            # return self.calculateFromAlpha(df_br_imputed, df_br_imputed['alpha_1'], df_br_imputed['alpha_2'])
        elif mode == 3:
            simpImp = SimpleImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE, strategy='mean')
            df_br_imputed = pd.DataFrame(simpImp.fit_transform(df_br), columns=df_br.columns)
            return df_br_imputed
        else:
            raise ValueError('Missing data mode not understood')

    def calculateFromAlpha(self, df_br, alpha_1, alpha_2):
        p_z_nu_1 = alpha_1*(df_br.pi_pz_1_br + df_br.pi0_pz_1_br)
        p_z_nu_2 = alpha_2*(df_br.pi_pz_2_br + df_br.pi0_pz_2_br)
        E_nu_1 = (self.m_tau**2 - (df_br.pi_E_1_br+df_br.pi0_E_1_br)**2 + (df_br.pi_pz_1_br + df_br.pi0_pz_1_br)
                  ** 2 + 2*p_z_nu_1*(df_br.pi_pz_1_br + df_br.pi0_pz_1_br))/(2*(df_br.pi_E_1_br+df_br.pi0_E_1_br))
        E_nu_2 = (self.m_tau**2 - (df_br.pi_E_2_br+df_br.pi0_E_2_br)**2 + (df_br.pi_pz_2_br + df_br.pi0_pz_2_br)
                  ** 2 + 2*p_z_nu_2*(df_br.pi_pz_2_br + df_br.pi0_pz_2_br))/(2*(df_br.pi_E_2_br+df_br.pi0_E_2_br))
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        df_br['E_nu_1'] = E_nu_1
        df_br['E_nu_2'] = E_nu_2
        df_br['p_t_nu_1'] = p_t_nu_1
        df_br['p_t_nu_2'] = p_t_nu_2
        df_br['p_z_nu_1'] = p_z_nu_1
        df_br['p_z_nu_2'] = p_z_nu_2
        return df_br
    
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
        -- Returns: alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 --
        Returns: df_br (with neutrino information contained)
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
        idx = alpha_1==NeutrinoReconstructor.DEFAULT_VALUE
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        p_t_nu_1[idx] = NeutrinoReconstructor.DEFAULT_VALUE
        p_t_nu_2[idx] = NeutrinoReconstructor.DEFAULT_VALUE
        # populate input df with neutrino variables
        df_br['alpha_1'] = alpha_1
        df_br['alpha_2'] = alpha_2
        df_br['E_nu_1'] = E_nu_1
        df_br['E_nu_2'] = E_nu_2
        df_br['p_t_nu_1'] = p_t_nu_1
        df_br['p_t_nu_2'] = p_t_nu_2
        df_br['p_z_nu_1'] = p_z_nu_1
        df_br['p_z_nu_2'] = p_z_nu_2
        # df_red = df[(df['alpha_1'] != NeutrinoReconstructor.DEFAULT_VALUE) & (df['alpha_2'] != NeutrinoReconstructor.DEFAULT_VALUE) & (
        #     df['E_nu_1'] != NeutrinoReconstructor.DEFAULT_VALUE) & (df['E_nu_2'] != NeutrinoReconstructor.DEFAULT_VALUE)].reset_index(drop=True)
        # AC.profileAlphaPz(df_red)
        # return alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2
        return df_br

    def runGraphs(self, df_reco_gen, termination=1000):
        """
        Creates profile graph and other graphs
        """
        load_alpha = True
        # df_reco_gen = self.loadRecoData(skip=load_alpha)
        df_br = self.loadBRGenData()
        AC = AlphaCalculator(df_reco_gen, df_br, self.binary, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed, default_value=NeutrinoReconstructor.DEFAULT_VALUE)
        alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = AC.runAlpha(termination=termination)
        p_t_nu_1 = np.sqrt(np.array(E_nu_1)**2 - np.array(p_z_nu_1)**2)
        p_t_nu_2 = np.sqrt(np.array(E_nu_2)**2 - np.array(p_z_nu_2)**2)
        p_t_nu_1[np.isnan(p_t_nu_1)] = NeutrinoReconstructor.DEFAULT_VALUE
        p_t_nu_2[np.isnan(p_t_nu_2)] = NeutrinoReconstructor.DEFAULT_VALUE
        df_br['alpha_1'] = alpha_1
        df_br['alpha_2'] = alpha_2
        df_br['E_nu_1'] = E_nu_1
        df_br['E_nu_2'] = E_nu_2
        df_br['p_t_nu_1'] = p_t_nu_1
        df_br['p_t_nu_2'] = p_t_nu_2
        df_br['p_z_nu_1'] = p_z_nu_1
        df_br['p_z_nu_2'] = p_z_nu_2
        df_red = df_br[(df_br['alpha_1'] != NeutrinoReconstructor.DEFAULT_VALUE)].reset_index(drop=True)
        print(f'Rejected {(len(df.index)-len(df_red.index))/len(df_red.index)*100:.4f}% events due to physical reasons')
        AC.profileAlphaPz(df_red, termination=termination)
        # AC.checkAlphaPz(df_red, termination=termination)

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
        "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
        "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
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
    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=True, termination=1000)
    # NR.runGraphs(df.reset_index(drop=True))
    df_inputs = NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=True, termination=1000)
    # print(df_inputs.head())
    print(NR.dealWithMissingData(df_inputs, mode=1).head())
    exit()
    # df_br['alpha_1'] = alpha_1
    # df_br['alpha_2'] = alpha_2
    # df_br['E_nu_1'] = E_nu_1
    # df_br['E_nu_2'] = E_nu_2
    # df_br['p_t_nu_1'] = p_t_nu_1
    # df_br['p_t_nu_2'] = p_t_nu_2
    # df_br['p_z_nu_1'] = p_z_nu_1
    # df_br['p_z_nu_2'] = p_z_nu_2
    # print(df_br.columns)
    pd.to_pickle(df_br, 'misc/df_br.pkl')
    # NR.test1(df.reset_index(drop=False), df_br, load_alpha=False, termination=1000)

    # THIS COMBINATION WORKS -> DON'T KNOW WHY
