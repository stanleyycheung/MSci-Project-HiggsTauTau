#%%
import numpy as np
import pandas as pd
import uproot
from pylorentz import Momentum4
import matplotlib.pyplot as plt
# %%
gen_path = "/vols/cms/dw515/masters_ntuples/MVAFILE_GEN_AllHiggs_tt.root"

tree_tt_gen = uproot.open(gen_path)['ntuple']
# %%
variables = [
    "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
    "dm_1", "dm_2",
    "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1",
    "pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1",
    "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1",
    "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
    "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2",
    "pi2_px_2", "pi2_py_2", "pi2_pz_2", "pi2_E_2",
    "pi3_px_2", "pi3_py_2", "pi3_pz_2", "pi3_E_2",
    "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2",
    "sv_x_1", "sv_y_1", "sv_z_1",
    "sv_x_2", "sv_y_2", "sv_z_2",
    "metx", 'mety',
    "nu_E_1", "nu_px_1", "nu_py_1", "nu_pz_1", 
    "nu_E_2", "nu_px_2", "nu_py_2", "nu_pz_2",]

df_gen = tree_tt_gen.pandas.df(variables)
df_gen.to_hdf('/vols/cms/shc3117/kinematics.h5', 'df')
#%%
df_gen = pd.read_hdf('/vols/cms/shc3117/kinematics.h5', 'df')
# %%
# choose a1_a1 channel
df = df_gen[(df_gen['dm_1']==10) & (df_gen['dm_2']==10)]

#%%
# rho_a1 channel
df = df_gen[(df_gen['dm_1']==1) & (df_gen['dm_2']==10)]
# %%
m_tau = 1.776
def getTheta(a1, sv):
    a1_p = np.c_[a1.p_x, a1.p_y, a1.p_z]
    a1_p_norm = a1_p/np.sqrt((a1_p ** 2).sum(-1))[..., np.newaxis]
    sv_norm = sv/np.sqrt((sv ** 2).sum(-1))[..., np.newaxis]
    theta = np.arccos(np.einsum('ij, ij->i', a1_p_norm, sv_norm))
    max_theta = np.arcsin((m_tau**2-a1.m**2)/(2*m_tau*a1.p))
    idx1 = max_theta<theta
    theta_f = theta
    theta_f[idx1] = max_theta[idx1]
    return theta_f, sv_norm

def ANSolution(m, p, theta):
    # p is the magnitude
    a = (m**2+m_tau**2)*p*np.cos(theta)
    d = ((m**2-m_tau**2)**2-4*m_tau**2*p**2*np.sin(theta)**2)
    d = np.round(d, 14) # for floating point error
    b = np.sqrt((m**2+p**2)*d)
    c = 2*(m**2+p**2*np.sin(theta)**2)
    return (a+b)/c, (a-b)/c

# %%
pi_2 = Momentum4(df['pi_E_2'], df['pi_px_2'], df['pi_py_2'], df['pi_pz_2'])
pi2_2 = Momentum4(df['pi2_E_2'], df['pi2_px_2'], df['pi2_py_2'], df['pi2_pz_2'])
pi3_2 = Momentum4(df['pi3_E_2'], df['pi3_px_2'], df['pi3_py_2'], df['pi3_pz_2'])
a1_2 = pi_2 + pi3_2 + pi2_2
nu_2 = Momentum4(df['nu_E_2'], df['nu_px_2'],  df['nu_py_2'], df['nu_pz_2'])
sv_2 = np.c_[df['sv_x_2'], df['sv_y_2'], df['sv_z_2']]
theta_f_2, sv_norm_2 = getTheta(a1_2, sv_2)
tau_sol_2 = ANSolution(a1_2.m, a1_2.p, theta_f_2)
# %%
tau_p_2_1 = tau_sol_2[0][:,None]*sv_norm_2
tau_p_2_2 = tau_sol_2[1][:,None]*sv_norm_2
E_tau_2_1 = np.sqrt(np.linalg.norm(tau_p_2_1, axis=1)**2 + m_tau**2)
E_tau_2_2 = np.sqrt(np.linalg.norm(tau_p_2_2, axis=1)**2 + m_tau**2)

tau_sol_2_1 = Momentum4(E_tau_2_1, *tau_p_2_1.T)
tau_sol_2_2 = Momentum4(E_tau_2_2, *tau_p_2_2.T)
nu_sol_2_1 = tau_sol_2_1 - a1_2
nu_sol_2_2 = tau_sol_2_2 - a1_2
# %%
# use met information to find rho neutrino

nu_x_sol_1_1 = df['metx'] - nu_sol_2_1[1]
nu_y_sol_1_1 = df['mety'] - nu_sol_2_1[2]


def getNu(nu_x, nu_y):
    pass



# %%
