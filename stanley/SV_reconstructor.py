# import uproot
import numpy as np
import scipy.stats as sps
import pandas as pd
# import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter
from pylorentz import Momentum4, Position4


class SVReconstructor:
    def __init__(self):
        self.save_dir = './sv_analysis'
        self.pickle_dir = './df_tt_gen.pkl'

    def loadData(self, channel='rho_rho'):
        df_tt = pd.read_pickle(self.pickle_dir)
        df = None
        if channel == 'rho_rho':
            df = df_tt[(df_tt['dm_1'] == 1) & (df_tt['dm_2'] == 1)]
        # TODO: To add other channels
        self.df = df.drop(["dm_1", "dm_2", "wt_cp_sm", "wt_cp_ps",
                           "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def constructDF(self):
        self.sv_df = self.df[['sv_x_1', 'sv_y_1',
                              'sv_z_1', 'sv_x_2', 'sv_y_2', 'sv_z_2']]
        self.pi_1, self.pi_2, self.pi0_1, self.pi0_2, self.rho_1, self.rho_2 = self.getRhoDecayProducts(
            self.df)
        self.sv_df['p_t_vis_1'] = self.rho_1.p
        self.sv_df['p_t_vis_2'] = self.rho_2.p

    def constructSV(self):
        x_fit_1, _ = self.profileplot(
            self.sv_df['p_t_vis_1'], self.sv_df['sv_x_1'], xlabel=r'$P_{T,1}^{vis}$', ylabel=r'$SV_1^x$', bins=150)
        y_fit_1, _ = self.profileplot(
            self.sv_df['p_t_vis_1'], self.sv_df['sv_y_1'], xlabel=r'$P_{T,1}^{vis}$', ylabel=r'$SV_1^y$', bins=150)
        z_fit_1, _ = self.profileplot(
            self.sv_df['p_t_vis_1'], self.sv_df['sv_z_1'], xlabel=r'$P_{T,1}^{vis}$', ylabel=r'$SV_1^z$', bins=150)
        fit_1 = np.vstack((x_fit_1, y_fit_1, z_fit_1))
        print(f"Fit in x for 1: {x_fit_1}")
        print(f"Fit in y for 1: {y_fit_1}")
        print(f"Fit in z for 1: {z_fit_1}")
        np.savetxt(f'{self.save_dir}/sv_fit_rho_rho_1.txt',
                   fit_1, delimiter=',')
        x_fit_2, _ = self.profileplot(
            self.sv_df['p_t_vis_2'], self.sv_df['sv_x_2'], xlabel=r'$P_{T,2}^{vis}$', ylabel=r'$SV_2^x$', bins=150)
        y_fit_2, _ = self.profileplot(
            self.sv_df['p_t_vis_2'], self.sv_df['sv_y_2'], xlabel=r'$P_{T,2}^{vis}$', ylabel=r'$SV_2^y$', bins=150)
        z_fit_2, _ = self.profileplot(
            self.sv_df['p_t_vis_2'], self.sv_df['sv_z_2'], xlabel=r'$P_{T,2}^{vis}$', ylabel=r'$SV_2^z$', bins=150)
        fit_2 = np.vstack((x_fit_2, y_fit_2, z_fit_2))
        print(f"Fit in x for 2: {x_fit_2}")
        print(f"Fit in y for 2: {y_fit_2}")
        print(f"Fit in z for 2: {z_fit_2}")
        np.savetxt(f'{self.save_dir}/sv_fit_rho_rho_2.txt',
                   fit_2, delimiter=',')
        return fit_1, fit_2

    def plotSV(self):
        sv_1 = np.sqrt(self.sv_df['sv_x_1']**2 +
                       self.sv_df['sv_y_1']**2 + self.sv_df['sv_z_1']**2)
        self.profileplot(self.sv_df['p_t_vis_1'], sv_1,
                         xlabel=r'$P_{T,1}^{vis}$', ylabel=r'$||SV_1||$', bins=150)
        plt.savefig(f'{self.save_dir}/profile_sv_rho_rho_1.png')
        # plt.show()
        sv_2 = np.sqrt(self.sv_df['sv_x_2']**2 +
                       self.sv_df['sv_y_2']**2 + self.sv_df['sv_z_2']**2)
        self.profileplot(self.sv_df['p_t_vis_2'], sv_2,
                         xlabel=r'$P_{T,2}^{vis}$', ylabel=r'$||SV_2||$', bins=150)
        plt.savefig(f'{self.save_dir}/profile_sv_rho_rho_2.png')
        plt.show()

    def run(self):
        print('Loading data')
        self.loadData()
        print('Constructin DF')
        self.constructDF()
        print('Construcing SV')
        self.constructSV()
        # self.plotSV()

    def profileplot(self, x, y, xlabel, ylabel, bins=100):
        means_result = sps.binned_statistic(
            x, [y, y**2], bins=bins, statistic='mean')
        means, means2 = means_result.statistic
        standard_deviations = np.sqrt(means2 - means**2)
        bin_edges = means_result.bin_edges
        bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
        # remove NaNs and single count bins
        nan_idx = np.argwhere(np.isnan(means)).flatten()
        zero_idx = np.argwhere(standard_deviations == 0)
        to_remove = np.union1d(nan_idx, zero_idx)
        means = np.delete(means, to_remove, None)
        bin_centers = np.delete(bin_centers, to_remove, None)
        standard_deviations = np.delete(standard_deviations, to_remove, None)
        count = Counter(means_result.binnumber)
        to_remove_set = set(to_remove)
        N = []
        for i in range(1, bins+1):
            if i-1 in to_remove_set:
                continue
            if i in count:
                N.append(count[i])
        # print(to_remove.shape)
        # print(bin_centers.shape, means.shape)
        yerr = standard_deviations/np.sqrt(N)
        # yerr = standard_deviations
        # fitting
        # print(bin_centers, means, yerr)
        fit, cov = np.polyfit(bin_centers, means, 1, w=1/yerr, cov=True)
        p = np.poly1d(fit)
        # print(f"Fit params: {fit[0]}, {fit[1]}")
        # print(f"Diag of cov: {cov[0][0]} , {cov[1][1]}")
        plt.figure()
        plt.errorbar(x=bin_centers, y=means, yerr=yerr,
                     linestyle='none', marker='.', capsize=2)
        plt.plot(bin_centers, p(bin_centers))
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        return fit, cov

    def getRhoDecayProducts(self, df_reco):
        pi_1 = Momentum4(df_reco['pi_E_1'], df_reco["pi_px_1"],
                         df_reco["pi_py_1"], df_reco["pi_pz_1"])
        pi_2 = Momentum4(df_reco['pi_E_2'], df_reco["pi_px_2"],
                         df_reco["pi_py_2"], df_reco["pi_pz_2"])
        pi0_1 = Momentum4(df_reco['pi0_E_1'], df_reco["pi0_px_1"],
                          df_reco["pi0_py_1"], df_reco["pi0_pz_1"])
        pi0_2 = Momentum4(df_reco['pi0_E_2'], df_reco["pi0_px_2"],
                          df_reco["pi0_py_2"], df_reco["pi0_pz_2"])
        rho_1 = pi_1 + pi0_1
        rho_2 = pi_2 + pi0_2
        return pi_1, pi_2, pi0_1, pi0_2, rho_1, rho_2


if __name__ == '__main__':
    SV = SVReconstructor()
    SV.run()
