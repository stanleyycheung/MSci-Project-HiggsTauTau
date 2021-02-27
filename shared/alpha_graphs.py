# %%
from NN import NeuralNetwork
from data_loader import DataLoader
from neutrino_reconstructor import NeutrinoReconstructor
import config
import uproot
from pylorentz import Momentum4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from utils import profileplot_plain, profileplot

# %%
# plotting alpha error curves
# NNet = NeuralNetwork(channel='rho_rho', gen=False, binary=True, write_filename='NN_output', show_graph=True)
# NNet.addons_config_reco['neutrino']['imputer_mode'] = 'pass'
# NNet.addons_config_reco['neutrino']['load_alpha'] = True
# print(NNet.addons_config_reco)
# df = NNet.initialize(NNet.addons_config_reco, read=False, from_hdf=True)
# # %%
# print(df.head())
# print(df.columns)
# print(df.shape)
# # %%
# # prepare gen level data
# variables = [
#     "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
#     "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
# ] + config.selectors_reco + config.particles_rho_rho
# DLoader = DataLoader(variables, 'rho_rho', gen=False)
# tree_tt = uproot.open(DataLoader.reco_root_path)["ntuple"]
# df_gen = tree_tt.pandas.df(variables)
# # %%
# df_clean, df_ps, df_sm = DLoader.cleanRecoData(df_gen)
# df_gen_b, y = DLoader.augmentDfToBinary(df_ps, df_sm)
# # %%
# print(df_gen_b.columns)
# print(df_gen_b.head())
# print(df_gen_b.shape)

# %%

extra_variables = [
    "gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1",  # leading neutrino, gen level
    "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2"  # subleading neutrino, gen level
]
DL = DataLoader(config.variables_rho_rho + extra_variables, 'rho_rho', False)
addons_config = {'neutrino': {'load_alpha': True, 'termination': 1000, 'imputer_mode': 'pass'},
                 'met': {},
                 'ip': {},
                 'sv': {}}
addons = addons_config.keys()
# df = DL.createRecoData(True, True, addons, addons_config)

df =  DL.loadRecoData(True, addons)
# %%
df = df[(df != 0).all(1)]
# %%
def rotationMatrixVectorised(vec1, vec2):
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


def boostAndRotateNeutrinos(df):
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
    rho_1 = pi_1 + pi0_1
    rho_1_boosted = rho_1.boost_particle(boost)
    rotationMatrices = rotationMatrixVectorised(rho_1_boosted[1:].T, np.tile(np.array([0, 0, 1]), (rho_1_boosted.e.shape[0], 1)))
    nu_1_boosted_rot = np.einsum('ij,ikj->ik', nu_1_boosted[1:].T, rotationMatrices)
    nu_2_boosted_rot = np.einsum('ij,ikj->ik', nu_2_boosted[1:].T, rotationMatrices)
    return np.c_[nu_1_boosted.e, nu_1_boosted_rot], np.c_[nu_2_boosted.e, nu_2_boosted_rot]


# %%
# boost neutrino data
# NReconstructor = NeutrinoReconstructor(True, 'rho_rho')
nu_1_boosted_rot, nu_2_boosted_rot = boostAndRotateNeutrinos(df)
# print(nu_1_boosted_rot, nu_2_boosted_rot)
nu_1_boosted_rot = Momentum4(*nu_1_boosted_rot.T)
nu_2_boosted_rot = Momentum4(*nu_2_boosted_rot.T)


# %%
plt.rcParams["figure.figsize"] = (10,8)
# p_z_error = nu_1_boosted_rot.p_z-df.p_z_nu_1
# plt.hist(np.clip(p_z_error, -200, 200), label=f'mean: {np.mean(p_z_error):.2f}\nstd dev: {np.std(p_z_error, ddof=1):.2f}', bins=200)
# plt.legend()
# plt.show()

# %%
p_z_error_1 = nu_1_boosted_rot.p_z-df.p_z_nu_1
p_z_error_2 = nu_2_boosted_rot.p_z-df.p_z_nu_2

plt.hist(np.clip(p_z_error_1, -100, 100), label=f'nu_1\nmean: {np.mean(p_z_error_1):.2f}\nstd dev: {np.std(p_z_error_1, ddof=1):.2f}', bins=200, alpha=0.5)
plt.hist(np.clip(p_z_error_2, -100, 100), label=f'nu_2\nmean: {np.mean(p_z_error_2):.2f}\nstd dev: {np.std(p_z_error_2, ddof=1):.2f}', bins=200, alpha=0.5)
plt.title('clipped between [-100, 100]')
plt.xlabel('error')
plt.ylabel('freq')
plt.legend()
plt.savefig('./alpha_analysis/error_graphs/error_rho_rho.png')
plt.show()
# %%
# plt.figure(figsize=(8,6))
p_z_rel_error_1 = p_z_error_1/nu_1_boosted_rot.p_z
p_z_rel_error_2 = p_z_error_2/nu_2_boosted_rot.p_z
plt.hist(np.clip(p_z_rel_error_1, -50, 5), label=f'nu_1\nmean: {np.mean(p_z_rel_error_1):.2f}\nstd dev: {np.std(p_z_rel_error_1, ddof=1):.2f}', bins=200, alpha=0.5)
plt.hist(np.clip(p_z_rel_error_2, -50, 5), label=f'nu_2\nmean: {np.mean(p_z_rel_error_2):.2f}\nstd dev: {np.std(p_z_rel_error_2, ddof=1):.2f}', bins=200, alpha=0.5)
plt.title('clipped between [-50, 5]')
plt.xlabel('relative error')
plt.ylabel('freq')
plt.legend()
plt.savefig('./alpha_analysis/error_graphs/rel_error_rho_rho.png')
plt.show()

# %%
import scipy.stats as sps
import numpy as np
from collections import Counter

def pplot(x, y, xlabel='', ylabel='', bins=100, mode=0):
    means_result = sps.binned_statistic(x, [y, y**2], bins=bins, statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    # remove NaNs and single count bins
    nan_idx = np.argwhere(np.isnan(means) ).flatten()
    zero_idx = np.argwhere(standard_deviations == 0)
    to_remove = np.union1d(nan_idx, zero_idx)
    means = np.delete(means, to_remove, None)
    bin_centers = np.delete(bin_centers, to_remove, None)
    standard_deviations = np.delete(standard_deviations, to_remove, None)
    count = Counter(means_result.binnumber)
    to_remove_set = set(to_remove)
    N = []
    for i in range(1,bins+1):
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
    # plt.figure()
    bin_centers = bin_centers[:-15]
    means = means[:-15]
    yerr = yerr[:-15]
    plt.errorbar(x=bin_centers, y=means, yerr=yerr, linestyle='none', marker='.', capsize=2)
    if mode == 1:
        fit, cov = np.polyfit(bin_centers, means, 1, w=1/yerr, cov=True)
        p = np.poly1d(fit)
        print(f"Fit params: {fit[0]}, {fit[1]}")
        print(f"Diag of cov: {cov[0][0]} , {cov[1][1]}")
        plt.plot(bin_centers, p(bin_centers), label=f'gradient:{fit[0]:.2f}+/-{cov[0][0]:.2f}\nintercept:{fit[1]:.3f}+/-{cov[1][1]:.2f}')
    # plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    # plt.tight_layout()

# %%
# plt.rcParams["figure.figsize"] = (8,6)
plt.figure()
pplot(df.alpha_1, p_z_error_1, xlabel='alpha', ylabel='p_z_error', bins=100, mode=1)
pplot(df.alpha_2, p_z_error_2, xlabel='alpha', ylabel='p_z_error', bins=100, mode=1)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_p_z_error.png')
plt.show()

# %%
plt.figure()
pplot(df.alpha_1, p_z_rel_error_1, xlabel='alpha', ylabel='p_z_rel_error', bins=100, mode=1)
pplot(df.alpha_2, p_z_rel_error_2, xlabel='alpha', ylabel='p_z_rel_error', bins=100, mode=1)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_p_z_re_error.png')
plt.show()

#%%
p_t_error_1 = nu_1_boosted_rot.p_t-df.p_t_nu_1
p_t_error_2 = nu_2_boosted_rot.p_t-df.p_t_nu_2
plt.figure()
pplot(df.alpha_1, p_t_error_1, xlabel='alpha', ylabel='p_t_error', bins=100, mode=0)
pplot(df.alpha_2, p_t_error_2, xlabel='alpha', ylabel='p_t_error', bins=100, mode=0)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_p_t_error.png')
plt.show()

#%%
p_t_rel_error_1 = p_t_error_1/nu_1_boosted_rot.p_t
p_t_rel_error_2 = p_t_error_2/nu_2_boosted_rot.p_t
plt.figure()
pplot(df.alpha_1, p_t_rel_error_1, xlabel='alpha', ylabel='p_t_rel_error', bins=100, mode=0)
pplot(df.alpha_2, p_t_rel_error_1, xlabel='alpha', ylabel='p_t_rel_error', bins=100, mode=0)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_p_t_rel_error.png')
plt.show()

#%%
E_error_1 = nu_1_boosted_rot.e-df.E_nu_1
E_error_2 = nu_2_boosted_rot.e-df.E_nu_2
plt.figure()
pplot(df.alpha_1, E_error_1, xlabel='alpha', ylabel='E_error', bins=100, mode=0)
pplot(df.alpha_2, E_error_2, xlabel='alpha', ylabel='E_error', bins=100, mode=0)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_E_error.png')
plt.show()

#%%
E_rel_error_1 = E_error_1/nu_1_boosted_rot.e
E_rel_error_1 = E_error_2/nu_2_boosted_rot.e
plt.figure()
pplot(df.alpha_1, E_rel_error_1, xlabel='alpha', ylabel='E_rel_error', bins=100, mode=0)
pplot(df.alpha_2, E_rel_error_1, xlabel='alpha', ylabel='E_rel_error', bins=100, mode=0)
plt.grid()
plt.savefig('./alpha_analysis/error_graphs/profile_E_rel_error.png')
plt.show()

# %%
print(df.alpha_1.describe())
print(df.alpha_2.describe())
# plt.hist(df.alpha_1)
# plt.show()


#%%
# data_points = np.c_[np.random.choice(nu_1_boosted_rot.p_z, 1000), np.random.choice(df.p_z_nu_1, 1000)]
# plt.plot(data_points.T[0], data_points.T[1], '.')
d = pd.DataFrame(np.c_[df.p_z_nu_1,nu_1_boosted_rot.p_z])
d = d[(d[0]<400) & (d[0]>-0) & (d[1]<100) & (d[1]>-0)]
plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(0, 100), np.linspace(0, 100), 'r')
# plt.xlim(-25,25)
plt.colorbar()
plt.xlabel('reco_p_z')
plt.ylabel('gen_p_z')
plt.savefig('./alpha_analysis/error_graphs/hexbin_p_z.png')
plt.show()
# %%
e = pd.DataFrame(np.c_[df.p_t_nu_1,nu_1_boosted_rot.p_t])
e = e[(e[0]<4) & (e[0]>-0) & (e[1]<4) & (e[1]>-0)]
plt.hexbin(e[0], e[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(0, 4), np.linspace(0, 4), 'r')
# plt.xlim(-25,25)
plt.colorbar()
plt.xlabel('reco_p_t')
plt.ylabel('gen_p_t')
plt.savefig('./alpha_analysis/error_graphs/hexbin_p_t.png')
plt.show()

# %%

f = pd.DataFrame(np.c_[df.E_nu_1,nu_1_boosted_rot.e])
f = f[(f[0]<300) & (f[0]>-0) & (f[1]<100) & (f[1]>-0)]
plt.hexbin(f[0], f[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(0, 100), np.linspace(0, 100), 'r')
# plt.xlim(-25,25)
plt.colorbar()
plt.xlabel('reco_E')
plt.ylabel('gen_E')
plt.savefig('./alpha_analysis/error_graphs/hexbin_E.png')
plt.show()
# %%
