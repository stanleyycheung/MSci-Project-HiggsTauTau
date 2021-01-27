import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
import matplotlib.pyplot as plt
from utils import profileplot, sps, profileplot_plain

class AlphaCalculator:
    # changed file paths to class variables
    pickle_dir = './df_tt_rho_rho.pkl'
    alpha_save_dir = './alpha_analysis'

    def __init__(self, df_reco, df_br, binary, m_higgs, m_tau, default_value, load=False, seed=1):
        np.random.seed(seed)
        self.m_higgs = m_higgs
        self.m_tau = m_tau
        self.binary = binary
        self.df = df_reco
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

    def runAlphaOld(self, termination=1000):
        if self.load:
            self.alpha_1 = np.load(f'{AlphaCalculator.alpha_save_dir}/alpha_1_{termination}.npy', allow_pickle=True)
            self.alpha_2 = np.load(f'{AlphaCalculator.alpha_save_dir}/alpha_2_{termination}.npy', allow_pickle=True)
            return self.alpha_1, self.alpha_2
        self.alpha_1, self.alpha_2 = [], []
        rejection = 0
        for i in range(self.df.shape[0]):
            E_miss_x_row = self.df.metx[i] 
            E_miss_y_row = self.df.mety[i] 
            rho_1_row = np.array([self.df.pi_E_1[i],self.df.pi_px_1[i],self.df.pi_py_1[i],self.df.pi_pz_1[i]])
            rho_2_row = np.array([self.df.pi_E_2[i],self.df.pi_px_2[i],self.df.pi_py_2[i],self.df.pi_pz_2[i]])
            row_mean = np.array([self.df.metx[i], self.df.mety[i]])
            row_cov = np.array(([self.df.metcov00[i],self.df.metcov01[i]],[self.df.metcov10[i],self.df.metcov11[i]]))
            alpha_1_loc, alpha_2_loc = self.getAlphaOld(E_miss_x_row, E_miss_y_row, rho_1_row, rho_2_row, row_mean, row_cov, termination=termination)
            self.alpha_1.append(alpha_1_loc)
            self.alpha_2.append(alpha_2_loc)
            if alpha_1_loc < 0:
                rejection += 1
            if i%100000 == 0:
                print(f'getting alpha for {i}, rejection: {rejection}/{self.df.shape[0]}')
        print('Saving alpha')
        np.save(f'{AlphaCalculator.alpha_save_dir}/alpha_1_{termination}.npy', self.alpha_1, allow_pickle=True)
        np.save(f'{AlphaCalculator.alpha_save_dir}/alpha_2_{termination}.npy', self.alpha_2, allow_pickle=True)
        return self.alpha_1, self.alpha_2


    def runAlpha(self, termination=1000):
        """
        Runs alpha calculation, and automatically saves them
        -- Return type: alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 --
        -- Changed to not return neutrino reconstructed info (15/01) --
        -- Returns: alpha_1, alpha_2 --
        Changed back return type to original (19/01)
        Return type: alpha_1, alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
        """
        if self.load:
            binary_str = ''
            if self.binary:
                binary_str += "_b"
            self.alpha_1 = np.load(f'{AlphaCalculator.alpha_save_dir}/alpha_1_{termination}'+binary_str+".npy", allow_pickle=True)
            self.alpha_2 = np.load(f'{AlphaCalculator.alpha_save_dir}/alpha_2_{termination}'+binary_str+".npy", allow_pickle=True)
            idx = self.alpha_1==self.DEFAULT_VALUE
            p_z_nu_1 = self.alpha_1*(self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br)
            p_z_nu_2 = self.alpha_2*(self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br)
            E_nu_1 = (self.m_tau**2 - (self.df_br.pi_E_1_br+self.df_br.pi0_E_1_br)**2 + (self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br)
                      ** 2 + 2*p_z_nu_1*(self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br))/(2*(self.df_br.pi_E_1_br+self.df_br.pi0_E_1_br))
            E_nu_2 = (self.m_tau**2 - (self.df_br.pi_E_2_br+self.df_br.pi0_E_2_br)**2 + (self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br)
                      ** 2 + 2*p_z_nu_2*(self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br))/(2*(self.df_br.pi_E_2_br+self.df_br.pi0_E_2_br))
            p_z_nu_1[idx] = self.DEFAULT_VALUE
            p_z_nu_2[idx] = self.DEFAULT_VALUE
            E_nu_1[idx] = self.DEFAULT_VALUE
            E_nu_2[idx] = self.DEFAULT_VALUE
            return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
            # return self.alpha_1, self.alpha_2
        self.alpha_1, self.alpha_2 = [], []
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = [], [], [], []
        rejection = 0
        E_miss_x_col = self.df.metx.to_numpy()
        E_miss_y_col = self.df.mety.to_numpy()
        pi_E_1_col = self.df.pi_E_1.to_numpy()
        pi_px_1_col = self.df.pi_px_1.to_numpy()
        pi_py_1_col = self.df.pi_py_1.to_numpy()
        pi_pz_1_col = self.df.pi_pz_1.to_numpy()
        pi_E_2_col = self.df.pi_E_2.to_numpy()
        pi_px_2_col = self.df.pi_px_2.to_numpy()
        pi_py_2_col = self.df.pi_py_2.to_numpy()
        pi_pz_2_col = self.df.pi_pz_2.to_numpy()
        metcov00_col = self.df.metcov00.to_numpy()
        metcov01_col = self.df.metcov01.to_numpy()
        metcov10_col = self.df.metcov10.to_numpy()
        metcov11_col = self.df.metcov11.to_numpy()
        for i in range(self.df.shape[0]):
            rho_1_row = np.array([pi_E_1_col[i], pi_px_1_col[i], pi_py_1_col[i], pi_pz_1_col[i]])
            rho_2_row = np.array([pi_E_2_col[i], pi_px_2_col[i], pi_py_2_col[i], pi_pz_2_col[i]])
            row_cov = np.array(([metcov00_col[i], metcov01_col[i]], [metcov10_col[i], metcov11_col[i]]))
            (alpha_1_loc, alpha_2_loc), (p_z_nu_1_loc, E_nu_1_loc, p_z_nu_2_loc, E_nu_2_loc) = self.getAlpha(i, E_miss_x_col[i], E_miss_y_col[i], rho_1_row, rho_2_row, row_cov, termination=termination)
            # alpha_1_loc, alpha_2_loc = self.getAlpha(i, E_miss_x_col[i], E_miss_y_col[i], rho_1_row, rho_2_row, row_cov, termination=termination)
            # alpha_1_loc, alpha_2_loc = self.getAlphaOld(E_miss_x_row, E_miss_y_row, rho_1_row, rho_2_row, row_mean, row_cov, termination=termination)
            p_z_nu_1.append(p_z_nu_1_loc)
            E_nu_1.append(E_nu_1_loc)
            p_z_nu_2.append(p_z_nu_2_loc)
            E_nu_2.append(E_nu_2_loc)
            self.alpha_1.append(alpha_1_loc)
            self.alpha_2.append(alpha_2_loc)
            if alpha_1_loc == self.DEFAULT_VALUE:
                rejection += 1
            if i%10000 == 0:
                print(f'getting alpha for {i}, rejection: {rejection}/{self.df.shape[0]}')
        print('Saving alpha')
        binary_str = ''
        if self.binary:
            binary_str += "_b"
        np.save(f'{AlphaCalculator.alpha_save_dir}/alpha_1_{termination}'+binary_str+".npy", self.alpha_1, allow_pickle=True)
        np.save(f'{AlphaCalculator.alpha_save_dir}/alpha_2_{termination}'+binary_str+".npy", self.alpha_2, allow_pickle=True)
        return self.alpha_1, self.alpha_2, p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2
        # return self.alpha_1, self.alpha_2

    def getAlphaOld(self, E_miss_x, E_miss_y, rho_1, rho_2, mean, cov, mode=1, termination=1000):
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
        if alpha_1 < 0 or alpha_2 < 0:
            E_miss_gen = np.random.multivariate_normal(mean, cov, termination)
        else:
            return alpha_1, alpha_2
        for i in range(termination):
            E_miss_x, E_miss_y = E_miss_gen[i]
            alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
            # print(E_miss_x, E_miss_y, alpha_1, alpha_2)
            if alpha_1 > 0 and alpha_2 > 0:
                return alpha_1, alpha_2
        return self.DEFAULT_VALUE, self.DEFAULT_VALUE

    def getAlpha(self, idx, E_miss_x, E_miss_y, rho_1, rho_2, cov, mode=1, termination=1000):
        """
        Calculates alpha with constraints, returns self.DEFAULT_VALUE if not possible
        Returns: (alpha_1, alpha_2), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
        """
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
        # p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1, alpha_2)
        # unpack rows
        pi_pz_1_br = self.df_br.pi_pz_1_br.to_numpy()
        pi_pz_2_br = self.df_br.pi_pz_2_br.to_numpy()
        pi0_pz_1_br = self.df_br.pi0_pz_1_br.to_numpy()
        pi0_pz_2_br = self.df_br.pi0_pz_2_br.to_numpy()
        pi_E_1_br = self.df_br.pi_E_1_br.to_numpy()
        pi_E_2_br = self.df_br.pi_E_2_br.to_numpy()
        pi0_E_1_br = self.df_br.pi0_E_1_br.to_numpy()
        pi0_E_2_br = self.df_br.pi0_E_2_br.to_numpy()
        mean = np.array([E_miss_x, E_miss_y])
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo2(idx, alpha_1, alpha_2, pi_pz_1_br, pi_pz_2_br, pi0_pz_1_br, pi0_pz_2_br, pi_E_1_br, pi_E_2_br, pi0_E_1_br, pi0_E_2_br)
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
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
        for i in range(termination):
            E_miss_x, E_miss_y = E_miss_gen[i]
            # print(E_miss_x, E_miss_y, alpha_1, alpha_2)
            # p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(idx, alpha_1[i], alpha_2[i])
            p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo2(idx, alpha_1[i], alpha_2[i], pi_pz_1_br, pi_pz_2_br, pi0_pz_1_br, pi0_pz_2_br, pi_E_1_br, pi_E_2_br, pi0_E_1_br, pi0_E_2_br)
            if alpha_1[i] > 0 and alpha_2[i] > 0 and np.abs(E_nu_1) > np.abs(p_z_nu_1) and np.abs(E_nu_2) > np.abs(p_z_nu_2) and E_nu_1 > 0 and E_nu_2 > 0:
            # if alpha_1[i] > 0 and alpha_2[i] > 0:
            # if alpha_1[i] > 0 and alpha_2[i] > 0 and np.abs(E_nu_1) > np.abs(p_z_nu_1) and np.abs(E_nu_2) > np.abs(p_z_nu_2):
                # return (alpha_1[i], alpha_2[i]), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
                return (alpha_1[i], alpha_2[i]), (p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2)
        # return (-1, -1), (-1, -1, -1, -1)
        # return self.DEFAULT_VALUE, self.DEFAULT_VALUE
        return (self.DEFAULT_VALUE, self.DEFAULT_VALUE), (self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE, self.DEFAULT_VALUE)

    def getReconstructedInfo(self, i, alpha_1, alpha_2):
        """
        Reconstructs the momenta of neutrinos in the BR frame
        """
        # print(self.df_br.pi_pz_1_br.iloc[i], i)
        # print(i, alpha_1, alpha_2)    
        p_z_nu_1 = alpha_1*(self.df_br.pi_pz_1_br.iloc[i] + self.df_br.pi0_pz_1_br.iloc[i])
        p_z_nu_2 = alpha_2*(self.df_br.pi_pz_2_br.iloc[i] + self.df_br.pi0_pz_2_br.iloc[i])
        E_nu_1 = (self.m_tau**2 - (self.df_br.pi_E_1_br.iloc[i] + self.df_br.pi0_E_1_br.iloc[i])**2 + (self.df_br.pi_pz_1_br.iloc[i] + self.df_br.pi0_pz_1_br.iloc[i])
                  ** 2 + 2*p_z_nu_1*(self.df_br.pi_pz_1_br.iloc[i] + self.df_br.pi0_pz_1_br.iloc[i]))/(2*(self.df_br.pi_E_1_br.iloc[i] + self.df_br.pi0_E_1_br.iloc[i]))
        E_nu_2 = (self.m_tau**2 - (self.df_br.pi_E_2_br.iloc[i] + self.df_br.pi0_E_2_br.iloc[i])**2 + (self.df_br.pi_pz_2_br.iloc[i] + self.df_br.pi0_pz_2_br.iloc[i])
                  ** 2 + 2*p_z_nu_2*(self.df_br.pi_pz_2_br.iloc[i] + self.df_br.pi0_pz_2_br.iloc[i]))/(2*(self.df_br.pi_E_2_br.iloc[i] + self.df_br.pi0_E_2_br.iloc[i]))
        return p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2

    def getReconstructedInfo2(self, i, alpha_1, alpha_2, pi_pz_1_br, pi_pz_2_br, pi0_pz_1_br, pi0_pz_2_br, pi_E_1_br, pi_E_2_br, pi0_E_1_br, pi0_E_2_br):
        p_z_nu_1 = alpha_1*(pi_pz_1_br[i] + pi0_pz_1_br[i])
        p_z_nu_2 = alpha_2*(pi_pz_2_br[i] + pi0_pz_2_br[i])
        E_nu_1 = (self.m_tau**2 - (pi_E_1_br[i] + pi0_E_1_br[i])**2 + (pi_pz_1_br[i] + pi0_pz_1_br[i])
                  ** 2 + 2*p_z_nu_1*(pi_pz_1_br[i] + pi0_pz_1_br[i]))/(2*(pi_E_1_br[i] + pi0_E_1_br[i]))
        E_nu_2 = (self.m_tau**2 - (pi_E_2_br[i] + pi0_E_2_br[i])**2 + (pi_pz_2_br[i] + pi0_pz_2_br[i])
                  ** 2 + 2*p_z_nu_2*(pi_pz_2_br[i] + pi0_pz_2_br[i]))/(2*(pi_E_2_br[i] + pi0_E_2_br[i]))
        return p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2

    def profileAlphaPz(self, df_red, termination=100):
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

    def calcAlpha(self, E_miss_x, E_miss_y, rho_1, rho_2, mode):
        # rhos are 4 vectors, not 3 vectors
        if mode == 1:
            alpha_2 = (E_miss_y*rho_1[1]-E_miss_x*rho_1[2])/(rho_2[2]*rho_1[1]-rho_2[1]*rho_1[2])
            alpha_1 = (E_miss_x - alpha_2*rho_2[1])/rho_1[1]
        elif mode == 2:
            alpha_1 = (E_miss_y*rho_2[1]-E_miss_x*rho_2[2])/(rho_1[2]*rho_2[1]-rho_1[1]*rho_2[2])
            alpha_2 = (self.m_higgs**2/2 - self.m_tau**2)/(rho_1[0]*rho_2[0]-rho_1[1]*rho_2[1]-rho_1[2]*rho_1[2]-rho_1[3]*rho_1[3])/(1+alpha_1) - 1
        elif mode == 3:
            alpha_2 = (E_miss_y*rho_1[1]-E_miss_x*rho_1[2])/(rho_2[2]*rho_1[1]-rho_2[1]*rho_1[2])
            alpha_1 = (self.m_higgs**2/2 - self.m_tau**2)/(rho_1[0]*rho_2[0]-rho_1[1]*rho_2[1]-rho_1[2]*rho_1[2]-rho_1[3]*rho_1[3])/(1+alpha_2) - 1
        else:
            raise ValueError('incorrect mode in parameters')
        return alpha_1, alpha_2 

