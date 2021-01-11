import numpy as np
from numpy.core.defchararray import upper
import pandas as pd
import matplotlib.pyplot as plt
from utils import profileplot, sps, profileplot_plain

class AlphaCalculator:
    def __init__(self, df_reco, df_br, m_higgs, m_tau, load=False, seed=1):
        np.random.seed(seed)
        self.m_higgs = m_higgs
        self.m_tau = m_tau
        self.pickle_dir = './df_tt_rho_rho.pkl'
        self.df = df_reco
        self.df_br = df_br
        self.load = load
        self.alpha_save_dir = '../stanley/alpha_analysis'

    def loadData(self, channel='rho_rho'):
        df_tt = pd.read_pickle(self.pickle_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['mva_dm_1'] == 1) & (df_tt['mva_dm_2'] == 1) & (
                df_tt["tau_decay_mode_1"] == 1) & (df_tt["tau_decay_mode_2"] == 1)]
        self.df = df.drop(["mva_dm_1", "mva_dm_2", "tau_decay_mode_1", "tau_decay_mode_2",
                           "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def runAlpha(self, termination=1000):
        if self.load:
            self.alpha_1 = np.load(f'{self.alpha_save_dir}/alpha_1_{termination}.npy', allow_pickle=True)
            self.alpha_2 = np.load(f'{self.alpha_save_dir}/alpha_2_{termination}.npy', allow_pickle=True)
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
            alpha_1_loc, alpha_2_loc = self.getAlpha(E_miss_x_row, E_miss_y_row, rho_1_row, rho_2_row, row_mean, row_cov, termination=termination)
            self.alpha_1.append(alpha_1_loc)
            self.alpha_2.append(alpha_2_loc)
            if alpha_1_loc < 0:
                rejection += 1
            if i%100000 == 0:
                print(f'getting alpha for {i}, rejection: {rejection}/{self.df.shape[0]}')
        print('Saving alpha')
        np.save(f'{self.alpha_save_dir}/alpha_1_{termination}.npy', self.alpha_1, allow_pickle=True)
        np.save(f'{self.alpha_save_dir}/alpha_2_{termination}.npy', self.alpha_2, allow_pickle=True)
        return self.alpha_1, self.alpha_2


    def getAlpha(self, E_miss_x, E_miss_y, rho_1, rho_2, mean, cov, mode=1, termination=1000):
        alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
        p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(alpha_1, alpha_2)
        if alpha_1 < 0 or alpha_2 < 0 or np.abs(E_nu_1) < np.abs(p_z_nu_1) or np.abs(E_nu_2) < np.abs(p_z_nu_2):
            E_miss_gen = np.random.multivariate_normal(mean, cov, termination)
        else:
            return alpha_1, alpha_2
        for i in range(termination):
            E_miss_x, E_miss_y = E_miss_gen[i]
            alpha_1, alpha_2 = self.calcAlpha(E_miss_x, E_miss_y, rho_1, rho_2, mode)
            # print(E_miss_x, E_miss_y, alpha_1, alpha_2)
            p_z_nu_1, E_nu_1, p_z_nu_2, E_nu_2 = self.getReconstructedInfo(alpha_1, alpha_2)
            if alpha_1 > 0 and alpha_2 > 0 or np.abs(E_nu_1) < np.abs(p_z_nu_1) or np.abs(E_nu_2) < np.abs(p_z_nu_2):
                return alpha_1, alpha_2
        return -1, -1

    def getReconstructedInfo(self, alpha_1, alpha_2):
        """
        Reconstructs the momenta of neutrinos in the BR frame
        """
        # doesnt work currently - returns an array
        p_z_nu_1 = alpha_1*(self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br)
        p_z_nu_2 = alpha_2*self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br
        E_nu_1 = (self.m_tau**2 - (self.df_br.pi_E_1_br+self.df_br.pi0_E_1_br)**2 + (self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br)
                  ** 2 + 2*p_z_nu_1*(self.df_br.pi_pz_1_br + self.df_br.pi0_pz_1_br))/(2*(self.df_br.pi_E_1_br+self.df_br.pi0_E_1_br))
        E_nu_2 = (self.m_tau**2 - (self.df_br.pi_E_2_br+self.df_br.pi0_E_2_br)**2 + (self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br)
                  ** 2 + 2*p_z_nu_2*(self.df_br.pi_pz_2_br + self.df_br.pi0_pz_2_br))/(2*(self.df_br.pi_E_2_br+self.df_br.pi0_E_2_br))
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
        plt.savefig(f'{self.alpha_save_dir}/err_profile_{termination}.png')
        profileplot(p_z_nu_1, rel_error, bins=500, xlabel='reconstructed p_z', ylabel='rel error', mode=0)
        plt.savefig(f'{self.alpha_save_dir}/relerr_profile_{termination}.png')
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
        plt.savefig(f'{self.alpha_save_dir}/relerr_{termination}.png')
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
        plt.savefig(f'{self.alpha_save_dir}/freq_{termination}.png')
        fig2, ax2 = plt.subplots(figsize=(8,6))
        rel_error = (gen_plot-reco_plot_1)/gen_plot
        # plt.hist(np.clip(rel_error, -1, 1), bins=500, label='termination=1000')
        upper_lim, lower_lim = 2, -2
        rel_error_red = rel_error[np.where((rel_error >= lower_lim) & (rel_error <= upper_lim))]
        ax2.hist(rel_error_red, bins=100, label=f'termination={termination}')
        ax2.set_ylabel('Freq')
        ax2.set_xlabel('Relative error')
        plt.savefig(f'{self.alpha_save_dir}/relerr_{termination}.png')
        fig3, ax3 = plt.subplots(figsize=(8,6))
        error = (gen_plot-reco_plot_1)
        ax3.hist(np.clip(error, -200, 200), bins=100)
        ax3.set_ylabel('Freq')
        ax3.set_xlabel('Error')
        plt.savefig(f'{self.alpha_save_dir}/err_{termination}.png')
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

