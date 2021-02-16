# import uproot
import numpy as np
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from pylorentz import Momentum4, Position4
from alpha_calculator import AlphaCalculator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
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

    def __init__(self, binary, channel, seed=config.seed_value):
        np.random.seed(seed)
        self.seed = seed
        self.binary = binary
        self.m_higgs = 125.18
        self.m_tau = 1.776
        self.channel = channel

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
        'flag - Simple flag
        'bayesian_ridge' - BayesianRidge algorithm
        'decision_tree' - DecisionTree algorithm
        'extra_trees' - similar to missForest in R
        'kn_reg' - similar to 'knn'
        'knn' - KNN algorithm
        'mean' - Replace with mean
        'remove' - Remove events
        """
        print('Imputing missing data')
        # change to return df_br and modify in place
        # alpha_flag = NeutrinoReconstructor.DEFAULT_VALUE + 1
        if mode == 'flag':
            df_br['flag'] = np.where(df_br['alpha_1']==NeutrinoReconstructor.DEFAULT_VALUE, 0, 1)
            return df_br
        elif mode == 'mean':
            simpImp = SimpleImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE, strategy='mean')
            df_br_imputed = pd.DataFrame(simpImp.fit_transform(df_br), columns=df_br.columns)
            return df_br_imputed
        elif mode == 'remove':
            df_br_red = df_br[df_br.alpha_1 != NeutrinoReconstructor.DEFAULT_VALUE]
            print(f'Reduced events from {df_br.shape[0]} to {df_br_red.shape[0]}')
            return df_br_red
        elif mode in {'bayesian_ridge', 'decision_tree', 'extra_trees', 'kn_reg', 'knn'}:
            # only leave rotated 4 vectors in df
            neutrino_features = ['alpha_1', 'alpha_2', 'E_nu_1', 'E_nu_2', 'p_t_nu_1', 'p_t_nu_2', 'p_z_nu_1', 'p_z_nu_2']
            if self.channel == 'rho_rho':
                features = [str(x)+'_br' for x in config.particles_rho_rho] + neutrino_features
            elif self.channel == 'rho_a1':
                features = [str(x)+'_br' for x in config.particles_rho_a1] + neutrino_features
            else:
                features = [str(x)+'_br' for x in config.particles_a1_a1] + neutrino_features
            df_br_red = df_br[features]
            print(df_br_red.columns)
            if mode == 'bayesian_ridge':
                # df_br['alpha_1'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
                # df_br['alpha_2'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
                # print(df_br.head())
                # default is BayesianRidge
                # itImp = IterativeImputer(missing_values=alpha_flag, random_state=0, verbose=1)
                itImp = IterativeImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE, random_state=config.seed_value, verbose=2, max_iter=10)
                df_br_imputed = pd.DataFrame(itImp.fit_transform(df_br_red), columns=df_br_red.columns)
                return df_br_imputed
                # return self.calculateFromAlpha(df_br_imputed, df_br_imputed['alpha_1'], df_br_imputed['alpha_2'])
            elif mode == 'decision_tree':
                itImp = IterativeImputer(estimator=DecisionTreeRegressor(max_features='sqrt', random_state=config.seed_value), missing_values=NeutrinoReconstructor.DEFAULT_VALUE, random_state=config.seed_value, verbose=2, max_iter=10)
                df_br_imputed = pd.DataFrame(itImp.fit_transform(df_br_red), columns=df_br_red.columns)
                return df_br_imputed
            elif mode == 'extra_trees':
                itImp = IterativeImputer(estimator=ExtraTreesRegressor(), missing_values=NeutrinoReconstructor.DEFAULT_VALUE, random_state=config.seed_value, verbose=2, max_iter=10)
                df_br_imputed = pd.DataFrame(itImp.fit_transform(df_br_red), columns=df_br_red.columns)
                return df_br_imputed
            elif mode == 'kn_reg':
                itImp = IterativeImputer(estimator=KNeighborsRegressor(n_neighbors=5), missing_values=NeutrinoReconstructor.DEFAULT_VALUE, random_state=config.seed_value, verbose=2, max_iter=10)
                df_br_imputed = pd.DataFrame(itImp.fit_transform(df_br_red), columns=df_br_red.columns)
            elif mode == 'knn':
                # df_br['alpha_1'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
                # df_br['alpha_2'].replace(NeutrinoReconstructor.DEFAULT_VALUE, alpha_flag, inplace=True)
                # KNNImp = KNNImputer(missing_values=alpha_flag, n_neighbors=2) 
                KNNImp = KNNImputer(missing_values=NeutrinoReconstructor.DEFAULT_VALUE)
                df_br_imputed = pd.DataFrame(KNNImp.fit_transform(df_br_red), columns=df_br_red.columns)
                return df_br_imputed
                # return self.calculateFromAlpha(df_br_imputed, df_br_imputed['alpha_1'], df_br_imputed['alpha_2'])
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

    def getPhiTau(self, df):
        def getPhiTauForOne(self, a1, sv):
            a1_p = np.c_[a1.p_x, a1.p_y, a1.p_z]
            a1_p_norm = a1_p/np.sqrt((a1_p ** 2).sum(-1))[..., np.newaxis]
            sv_norm = sv/np.sqrt((sv ** 2).sum(-1))[..., np.newaxis]
            theta = np.arccos(np.einsum('ij, ij->i', a1_p_norm, sv_norm))
            max_theta = np.arcsin((self.m_tau**2-a1.m**2)/(2*self.m_tau*a1.p))
            idx1 = max_theta<theta
            theta_f = theta
            theta_f[idx1] = max_theta[idx1]
            return theta_f, sv_norm
        
        if self.channel == 'rho_a1':
            # TODO: not completed
            pi_1 = Momentum4(df['pi_E_1'], df['pi_px_1'], df['pi_py_1'], df['pi_pz_1'])
            pi0_1 = Momentum4(df['pi0_E_1'], df['pi0_px_1'], df['pi0_py_1'], df['pi0_pz_1'])
            pi_2 = Momentum4(df['pi_E_2'], df['pi_px_2'], df['pi_py_2'], df['pi_pz_2'])
            pi2_2 = Momentum4(df['pi2_E_2'], df['pi2_px_2'], df['pi2_py_2'], df['pi2_pz_2'])
            pi3_2 = Momentum4(df['pi3_E_2'], df['pi3_px_2'], df['pi3_py_2'], df['pi3_pz_2'])            
            metx, mety = df['metx'], df['mety']
            a1_2 = pi_2 + pi3_2 + pi2_2
            sv_2 = np.c_[df['sv_x_2'], df['sv_y_2'], df['sv_z_2']]
            theta_f_2 = getPhiTauForOne(a1_2, sv_2)
            sol_2, sv_norm_2 = self.ANSolution(a1_2.m, a1_2.p, theta_f_2)
            tau_p_2_1 = sol_2[0][:,None]*sv_norm_2
            tau_p_2_2 = sol_2[1][:,None]*sv_norm_2
            E_tau_2_1 = np.sqrt(np.linalg.norm(tau_p_2_1, axis=1)**2 + self.m_tau**2)
            E_tau_2_2 = np.sqrt(np.linalg.norm(tau_p_2_2, axis=1)**2 + self.m_tau**2)
        
        elif self.channel == "a1_a1":
            pi_1 = Momentum4(df['pi_E_1'], df['pi_px_1'], df['pi_py_1'], df['pi_pz_1'])
            pi2_1 = Momentum4(df['pi2_E_1'], df['pi2_px_1'], df['pi2_py_1'], df['pi2_pz_1'])
            pi3_1 = Momentum4(df['pi3_E_1'], df['pi3_px_1'], df['pi3_py_1'], df['pi3_pz_1'])
            pi_2 = Momentum4(df['pi_E_2'], df['pi_px_2'], df['pi_py_2'], df['pi_pz_2'])
            pi2_2 = Momentum4(df['pi2_E_2'], df['pi2_px_2'], df['pi2_py_2'], df['pi2_pz_2'])
            pi3_2 = Momentum4(df['pi3_E_2'], df['pi3_px_2'], df['pi3_py_2'], df['pi3_pz_2'])
            a1_1 = pi_1 + pi3_1 + pi2_1
            a1_2 = pi_2 + pi3_2 + pi2_2
            sv_1 = np.c_[df['sv_x_1'], df['sv_y_1'], df['sv_z_1']]
            sv_2 = np.c_[df['sv_x_2'], df['sv_y_2'], df['sv_z_2']]
            theta_f_1 = getPhiTauForOne(a1_1, sv_1)
            theta_f_2 = getPhiTauForOne(a1_2, sv_2)
            sol_1, sv_norm_1 = self.ANSolution(a1_1.m, a1_1.p, theta_f_1)
            sol_2, sv_norm_2 = self.ANSolution(a1_2.m, a1_2.p, theta_f_2)
            tau_p_1_1 = sol_1[0][:,None]*sv_norm_1
            tau_p_1_2 = sol_1[1][:,None]*sv_norm_1
            tau_p_2_1 = sol_2[0][:,None]*sv_norm_2
            tau_p_2_2 = sol_2[1][:,None]*sv_norm_2
            E_tau_1_1 = np.sqrt(np.linalg.norm(tau_p_1_1, axis=1)**2 + self.m_tau**2)
            E_tau_1_2 = np.sqrt(np.linalg.norm(tau_p_1_2, axis=1)**2 + self.m_tau**2)
            E_tau_2_1 = np.sqrt(np.linalg.norm(tau_p_2_1, axis=1)**2 + self.m_tau**2)
            E_tau_2_2 = np.sqrt(np.linalg.norm(tau_p_2_2, axis=1)**2 + self.m_tau**2)
            tau_1_1 = Momentum4(E_tau_1_1, *tau_p_1_1.T)
            tau_1_2 = Momentum4(E_tau_1_2, *tau_p_1_2.T)
            tau_2_1 = Momentum4(E_tau_2_1, *tau_p_2_1.T)
            tau_2_2 = Momentum4(E_tau_2_2, *tau_p_2_2.T)
            rest_frame = a1_1 + a1_2
            boost = Momentum4(rest_frame[0], -rest_frame[1], -rest_frame[2], -rest_frame[3])
            tau_1_1_boosted = tau_1_1.boost_particle(boost)
            tau_1_2_boosted = tau_1_2.boost_particle(boost)
            tau_2_1_boosted = tau_2_1.boost_particle(boost)
            tau_2_2_boosted = tau_2_2.boost_particle(boost)
            tau_p_1_1_b_norm = tau_1_1_boosted[1:].T/np.linalg.norm(tau_1_1_boosted[1:].T, axis=1)[:,None]
            tau_p_1_2_b_norm = tau_1_2_boosted[1:].T/np.linalg.norm(tau_1_2_boosted[1:].T, axis=1)[:,None]
            tau_p_2_1_b_norm = tau_2_1_boosted[1:].T/np.linalg.norm(tau_2_1_boosted[1:].T, axis=1)[:,None]
            tau_p_2_2_b_norm = tau_2_2_boosted[1:].T/np.linalg.norm(tau_2_2_boosted[1:].T, axis=1)[:,None]
            angle_1_1 = np.einsum('ij, ij->i', tau_p_1_1_b_norm, tau_p_2_1_b_norm)
            angle_1_2 = np.einsum('ij, ij->i', tau_p_1_1_b_norm, tau_p_2_2_b_norm)
            angle_2_1 = np.einsum('ij, ij->i', tau_p_1_2_b_norm, tau_p_2_1_b_norm)
            angle_2_2 = np.einsum('ij, ij->i', tau_p_1_2_b_norm, tau_p_2_2_b_norm)
            return angle_1_1, angle_1_2, angle_2_1, angle_2_2

    def ANSolution(self, m, p, theta):
        # p is the magnitude
        a = (m**2+self.m_tau**2)*p*np.cos(theta)
        d = ((m**2-self.m_tau**2)**2-4*self.m_tau**2*p**2*np.sin(theta)**2)
        d = np.round(d, 14) # for floating point error
        b = np.sqrt((m**2+p**2)*d)
        c = 2*(m**2+p**2*np.sin(theta)**2)
        return (a+b)/c, (a-b)/c

    def runAlphaReconstructor(self, df_reco, df_br, load_alpha, termination=1000):
        """
        Calculates the alphas and reconstructs neutrino momenta
        df_reco - events straight from the .root file
        df_br - BR events
        Default error value: -1
        To do:
        - include azimuthal angles of the neutrinos
        Notes:
        - not inserting four-vector -> other components are just 0
        -- Returns: alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 --
        Returns: df_br (with neutrino information contained)
        """
        AC = AlphaCalculator(self.channel, df_reco, df_br, self.binary, self.m_higgs,
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

    def runGenAlphaReconstructor(self, df_gen, df_br, load_alpha):
        AC = AlphaCalculator(self.channel, df_gen, df_br, self.binary, self.m_higgs,
                             self.m_tau, load=load_alpha, seed=self.seed, default_value=NeutrinoReconstructor.DEFAULT_VALUE)
        alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = AC.runAlphaGen()
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

    def test1(self, df_reco, df_br, load_alpha, termination=1000):
        """
        Runs test from old criteria
        """
        # df_reco = self.loadRecoData(skip=load_alpha)
        AC = AlphaCalculator(df_reco, df_br, self.binary, self.m_higgs,
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
    import config
    gen = False
    channel = 'rho_rho'
    NR = NeutrinoReconstructor(binary=True, channel=channel)
    if not gen:
        addons_config_reco = {'neutrino': {'load_alpha':False, 'termination':100}, 'met': {}, 'ip': {}, 'sv': {}}
        addons = addons_config_reco.keys()
        DL = DataLoader(config.variables_rho_rho, channel, gen)
        df, df_ps, df_sm = DL.cleanRecoData(DL.readRecoData(from_hdf=True))
        df_br = DL.loadRecoData(True, addons).reset_index(drop=True)
    else:
        addons_config_gen = {'neutrino': {'load_alpha':False, 'termination':100}, 'sv': {}}
        addons = addons_config_gen.keys()
        DL = DataLoader(config.variables_gen_rho_rho, channel, gen)
        df, df_ps, df_sm = DL.cleanGenData(DL.readGenData(from_hdf=True))
        df_br = DL.loadGenData(True, addons).reset_index(drop=True)
    df_b, _ = DL.augmentDfToBinary(df_ps, df_sm)
    # NR.runAlphaReconstructor(df_b.reset_index(drop=True), df_br, load_alpha=False, termination=100)
    df_br_imputed = NR.dealWithMissingData(df_br, mode='extra_trees')
    print(df_br_imputed.head())
    # NR = NeutrinoReconstructor(binary=False, channel='rho_rho')
    # # NR.testRunAlphaReconstructor()
    # variables_rho_rho = [
    #     "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    #     "aco_angle_1",
    #     "mva_dm_1", "mva_dm_2",
    #     "tau_decay_mode_1", "tau_decay_mode_2",
    #     "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    #     "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    #     "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    #     "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
    #     "y_1_1", "y_1_2",
    #     'met', 'metx', 'mety',
    #     'metcov00', 'metcov01', 'metcov10', 'metcov11',
    #     "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
    #     "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
    # ]
    # channel = 'rho_rho'
    # DL = DataLoader(variables_rho_rho, channel, gen=False)
    # df, df_rho_ps, df_rho_sm = DL.cleanRecoData(DL.readRecoData(from_pickle=True))
    # df_br = DL.loadRecoData(binary=False).reset_index(drop=True)
    # # augment the binary df
    # df_reco, _ = DL.augmentDfToBinary(df_rho_ps, df_rho_sm)
    # slightly different lengths - due to binary/non_binary

    # debug = pd.read_pickle('./misc/debugging_2.pkl')
    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(debug, debug, load_alpha=False, termination=100)
    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(df_reco_gen, df_br, load_alpha=False, termination=100)
    # alpha_1, alpha_2, E_nu_1, E_nu_2, p_t_nu_1, p_t_nu_2, p_z_nu_1, p_z_nu_2 = NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=True, termination=1000)
    # NR.runGraphs(df.reset_index(drop=True))
    # df_inputs = NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=True, termination=1000)
    # print(df_inputs.head())
    # print(NR.dealWithMissingData(df_inputs, mode=1).head())

    # NR.runAlphaReconstructor(df.reset_index(drop=True), df_br, load_alpha=False, termination=100)
    # print(df.shape, df_br.shape)
    # exit()
    # df_br['alpha_1'] = alpha_1
    # df_br['alpha_2'] = alpha_2
    # df_br['E_nu_1'] = E_nu_1
    # df_br['E_nu_2'] = E_nu_2
    # df_br['p_t_nu_1'] = p_t_nu_1
    # df_br['p_t_nu_2'] = p_t_nu_2
    # df_br['p_z_nu_1'] = p_z_nu_1
    # df_br['p_z_nu_2'] = p_z_nu_2
    # print(df_br.columns)
    # pd.to_pickle(df_br, 'misc/df_br.pkl')
    # NR.test1(df.reset_index(drop=False), df_br, load_alpha=False, termination=1000)

    # THIS COMBINATION WORKS -> DON'T KNOW WHY
