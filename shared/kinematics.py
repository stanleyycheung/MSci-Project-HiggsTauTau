# %%
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
    "nu_E_2", "nu_px_2", "nu_py_2", "nu_pz_2", ]

df_gen = tree_tt_gen.pandas.df(variables)
df_gen.to_hdf('/vols/cms/shc3117/kinematics.h5', 'df')
# %%
df_gen = pd.read_hdf('/vols/cms/shc3117/kinematics.h5', 'df')
# %%
# choose a1_a1 channel
# df = df_gen[(df_gen['dm_1']==10) & (df_gen['dm_2']==10)]

# %%
# rho_a1 channel
# clean of nans
df = df_gen[(df_gen['dm_1'] == 1) & (df_gen['dm_2'] == 10)]
df = df.drop(["pi2_E_1", "pi2_px_1", "pi2_py_1", "pi2_pz_1", "pi3_E_1", "pi3_px_1", "pi3_py_1", "pi3_pz_1", "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2"], axis=1)
df = df.dropna()
df = df[(df != 0).all(1)]
# %%
m_tau = 1.776


def getTheta(a1, sv):
    a1_p = np.c_[a1.p_x, a1.p_y, a1.p_z]
    a1_p_norm = a1_p/np.sqrt((a1_p ** 2).sum(-1))[..., np.newaxis]
    sv_norm = sv/np.sqrt((sv ** 2).sum(-1))[..., np.newaxis]
    theta = np.arccos(np.einsum('ij, ij->i', a1_p_norm, sv_norm))
    max_theta = np.arcsin((m_tau**2-a1.m**2)/(2*m_tau*a1.p))
    idx1 = max_theta < theta
    theta_f = theta
    theta_f[idx1] = max_theta[idx1]
    return theta_f, sv_norm


def ANSolution(m, p, theta):
    # p is the magnitude
    a = (m**2+m_tau**2)*p*np.cos(theta)
    d = ((m**2-m_tau**2)**2-4*m_tau**2*p**2*np.sin(theta)**2)
    d = np.round(d, 14)  # for floating point error
    b = np.sqrt((m**2+p**2)*d)
    c = 2*(m**2+p**2*np.sin(theta)**2)
    return (a+b)/c, (a-b)/c


# %%
pi_1 = Momentum4(df['pi_E_1'], df['pi_px_1'], df['pi_py_1'], df['pi_pz_1'])
pi0_1 = Momentum4(df['pi0_E_1'], df['pi0_px_1'], df['pi0_py_1'], df['pi0_pz_1'])
rho_1 = pi_1 + pi0_1
pi_2 = Momentum4(df['pi_E_2'], df['pi_px_2'], df['pi_py_2'], df['pi_pz_2'])
pi2_2 = Momentum4(df['pi2_E_2'], df['pi2_px_2'], df['pi2_py_2'], df['pi2_pz_2'])
pi3_2 = Momentum4(df['pi3_E_2'], df['pi3_px_2'], df['pi3_py_2'], df['pi3_pz_2'])
a1_2 = pi_2 + pi3_2 + pi2_2
nu_2 = Momentum4(df['nu_E_2'], df['nu_px_2'],  df['nu_py_2'], df['nu_pz_2'])
sv_2 = np.c_[df['sv_x_2'], df['sv_y_2'], df['sv_z_2']]
theta_f_2, sv_norm_2 = getTheta(a1_2, sv_2)
tau_sol_2 = ANSolution(a1_2.m, a1_2.p, theta_f_2)

# %%
tau_p_2_1 = tau_sol_2[0][:, None]*sv_norm_2
tau_p_2_2 = tau_sol_2[1][:, None]*sv_norm_2
E_tau_2_1 = np.sqrt(np.linalg.norm(tau_p_2_1, axis=1)**2 + m_tau**2)
E_tau_2_2 = np.sqrt(np.linalg.norm(tau_p_2_2, axis=1)**2 + m_tau**2)

tau_sol_2_1 = Momentum4(E_tau_2_1, *tau_p_2_1.T)
tau_sol_2_2 = Momentum4(E_tau_2_2, *tau_p_2_2.T)
nu_sol_2_1 = tau_sol_2_1 - a1_2
nu_sol_2_2 = tau_sol_2_2 - a1_2
# %%
# use met information to find rho neutrino

nu_x_sol_1_1 = (df['metx'] - nu_sol_2_1[1]).to_numpy()
nu_y_sol_1_1 = (df['mety'] - nu_sol_2_1[2]).to_numpy()
nu_x_sol_1_2 = (df['metx'] - nu_sol_2_2[1]).to_numpy()
nu_y_sol_1_2 = (df['mety'] - nu_sol_2_2[2]).to_numpy()


def getNuPz(had, nu_p_x, nu_p_y):
    a = (1-had.p_z**2)
    b = -2*((m_tau**2 - had.m**2)/(2*had.e)*had.p_z + had.p_y*nu_p_y*had.p_z + had.p_x*nu_p_x*had.p_z)
    c = nu_p_x**2 + nu_p_y**2 - ((m_tau**2 - had.m**2)/(2*had.e))**2 - (had.p_x*nu_p_x)**2 - (had.p_y*nu_p_y)**2 - 2 * \
        ((m_tau**2 - had.m**2)/(2*had.e)*(had.p_x*nu_p_x+had.p_y*nu_p_y) + had.p_x*nu_p_x*had.p_y*nu_p_y)
    # return b**2-4*a*c
    disc = b**2-4*a*c
    disc = np.where(disc < 0, 0, disc)
    # return disc
    return (-b+np.sqrt(disc))/(2*a), (-b-np.sqrt(disc))/(2*a)


def calcNuE(nu_p_x, nu_p_y, nu_p_z):
    return np.sqrt(nu_p_x**2 + nu_p_y**2 + nu_p_z**2)


def getNuE(had, nu_p_x, nu_p_y):
    a = (had.e**2-had.p_z**2)
    b = -2*(((m_tau**2 - had.m**2)/2)*had.e + had.e*(had.p_x*nu_p_x+had.p_y*nu_p_y))
    c = ((m_tau**2 - had.m**2)/2)**2 + (had.p_x*nu_p_x)**2 + (had.p_y*nu_p_y)**2 + (m_tau**2-had.m**2) * \
        (had.p_x*nu_p_x+had.p_y*nu_p_y) + 2*had.p_x*nu_p_x*had.p_y*nu_p_y + had.p_z**2*nu_p_x**2 + had.p_z**2*nu_p_y
    disc = b**2-4*a*c
    disc = np.where(disc < 0, 0, disc)
    return (-b+np.sqrt(disc))/(2*a), (-b-np.sqrt(disc))/(2*a)

# %%


nu_p_z_1_1_1, nu_p_z_1_1_2 = getNuPz(rho_1, nu_x_sol_1_1, nu_y_sol_1_1)
# E_nu_1, E_nu_2 = getNuE(rho_1, nu_x_sol_1_1, nu_y_sol_1_1)
E_nu_1_1_1 = calcNuE(nu_x_sol_1_1, nu_y_sol_1_1, nu_p_z_1_1_1)
E_nu_1_1_2 = calcNuE(nu_x_sol_1_1, nu_y_sol_1_1, nu_p_z_1_1_2)

nu_p_z_1_2_1, nu_p_z_1_2_2 = getNuPz(rho_1, nu_x_sol_1_2, nu_y_sol_1_2)
# E_nu_1, E_nu_2 = getNuE(rho_1, nu_x_sol_1_1, nu_y_sol_1_1)
E_nu_1_2_1 = calcNuE(nu_x_sol_1_2, nu_y_sol_1_2, nu_p_z_1_2_1)
E_nu_1_2_2 = calcNuE(nu_x_sol_1_2, nu_y_sol_1_2, nu_p_z_1_2_2)

nu_1_1_1 = Momentum4(E_nu_1_1_1, nu_x_sol_1_1, nu_y_sol_1_1, nu_p_z_1_1_1)
nu_1_1_2 = Momentum4(E_nu_1_1_2, nu_x_sol_1_1, nu_y_sol_1_1, nu_p_z_1_1_2)
nu_1_2_1 = Momentum4(E_nu_1_2_1, nu_x_sol_1_2, nu_y_sol_1_2, nu_p_z_1_2_1)
nu_1_2_2 = Momentum4(E_nu_1_2_2, nu_x_sol_1_2, nu_y_sol_1_2, nu_p_z_1_2_2)

# %%

# now use higgs mass constraint
# first particle taus
tau_1_1_1 = rho_1 + nu_1_1_1
tau_1_1_2 = rho_1 + nu_1_1_2
tau_1_2_1 = rho_1 + nu_1_2_1
tau_1_2_2 = rho_1 + nu_1_2_2
# second particle taus
# tau_sol_2_1, tau_sol_2_2

taus_1 = [tau_1_1_1, tau_1_1_2, tau_1_2_1, tau_1_2_2]
taus_2 = [tau_sol_2_1, tau_sol_2_2]

# higgs = []
higgs_combinations = [(tau_1_1_1, tau_sol_2_1), (tau_1_1_1, tau_sol_2_2),
                      (tau_1_1_2, tau_sol_2_1), (tau_1_1_2, tau_sol_2_2),
                      (tau_1_2_1, tau_sol_2_1), (tau_1_2_1, tau_sol_2_2),
                      (tau_1_2_2, tau_sol_2_1), (tau_1_2_2, tau_sol_2_2),]
# for tau_1 in taus_1:
#     for tau_2 in taus_2:
#         higgs.append(tau_1+tau_2)
higgs = [x+y for x,y in higgs_combinations]
higgs_mass = np.array([x.m for x in higgs])

# %%
for h in higgs:
    h_mass = np.clip(h.m, 0, 800)
    plt.hist(h_mass, alpha=0.5, bins=100)
plt.axvline(x=125.35, c='r')
plt.xlabel('mass (GeV)')
plt.savefig('./kinematic_graphs/higgs_combination_mass.PNG')
plt.show()
# %%
closest_higgs_idx = np.argmin(np.abs(higgs_mass.T-125.35), axis=1)
closest_higgs_mass = higgs_mass.T[np.arange(0, len(closest_higgs_idx)), closest_higgs_idx]
# %%
closest_neutrino_pair = []
for i, idx in enumerate(closest_higgs_idx):
    neutrino_pair = higgs_combinations[idx]
    first_neutrino = [neutrino_pair[0].e[i], neutrino_pair[0].p_x[i], neutrino_pair[0].p_y[i], neutrino_pair[0].p_z[i]]
    second_neutrino = [neutrino_pair[1].e[i], neutrino_pair[1].p_x[i], neutrino_pair[1].p_y[i], neutrino_pair[1].p_z[i]]
    closest_neutrino_pair.append([first_neutrino, second_neutrino])
    # if i%10000:
closest_neutrino_pair = np.array(closest_neutrino_pair)
# %%
est_nu_1 = Momentum4(*closest_neutrino_pair[:,0].T)
est_nu_2 = Momentum4(*closest_neutrino_pair[:,1].T)
# %%
plt.hist(np.clip(closest_higgs_mass, 0, 800), bins=100)
plt.axvline(x=125.35, c='r')
plt.xlabel('mass (GeV)')
plt.savefig('./kinematic_graphs/higgs_closest_mass.PNG')
plt.show()
# %%
# compare to actual gen neutrinos
gen_nu_1 = Momentum4(df['nu_E_1'], df['nu_px_1'], df['nu_py_1'], df['nu_pz_1'])
# gen_nu_2 = Momentum4(df['nu_E_2'], df['nu_px_2'], df['nu_py_2'], df['nu_pz_2'])

# %%
plt.rcParams["figure.figsize"] = (8,6)
# %%
d = pd.DataFrame(np.c_[est_nu_1.p_x, gen_nu_1.p_x])
d = d[(d[0]<300) & (d[0]>-300) & (d[1]<200) & (d[1]>-200)]
plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(-200, 200), np.linspace(-200, 200), 'r')
plt.colorbar()
plt.xlabel('est_p_x')
plt.ylabel('gen_p_x')
plt.savefig('./kinematic_graphs/hex_p_x.PNG')
plt.show()

# %%
d = pd.DataFrame(np.c_[est_nu_1.p_y, gen_nu_1.p_y])
d = d[(d[0]<300) & (d[0]>-300) & (d[1]<200) & (d[1]>-200)]
plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(-200, 200), np.linspace(-200, 200), 'r')
plt.colorbar()
plt.xlabel('est_p_y')
plt.ylabel('gen_p_y')
plt.savefig('./kinematic_graphs/hex_p_y.PNG')
plt.show()
# %%
# %%
d = pd.DataFrame(np.c_[est_nu_1.p_z, gen_nu_1.p_z])
d = d[(d[0]<3000) & (d[0]>-3000) & (d[1]<1000) & (d[1]>-1000)]
plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(-1000, 1000), np.linspace(-1000, 1000), 'r')
plt.colorbar()
plt.xlabel('est_p_z')
plt.ylabel('gen_p_z')
plt.savefig('./kinematic_graphs/hex_p_z.PNG')
plt.show()

# %%
d = pd.DataFrame(np.c_[est_nu_1.e, gen_nu_1.e])
d = d[(d[0]<2000) & (d[0]>-0) & (d[1]<600) & (d[1]>-0)]
plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
plt.plot(np.linspace(-0, 600), np.linspace(-0, 600), 'r')
plt.colorbar()
plt.xlabel('est_E')
plt.ylabel('gen_E')
plt.savefig('./kinematic_graphs/hex_E.PNG')
plt.show()
# %%
error_p_x = gen_nu_1.p_x - est_nu_1.p_x
error_p_y = gen_nu_1.p_y - est_nu_1.p_y
error_p_z = gen_nu_1.p_z - est_nu_1.p_z
error_E = gen_nu_1.e - est_nu_1.e
error_p = gen_nu_1.p - est_nu_1.p
plt.hist(np.clip(error_p_x, -500, 500), bins=200, alpha=0.5, label=f'p_x\nmean:{np.mean(error_p_x):.2f}\nstd:{np.std(error_p_x, ddof=1):.2f}')
plt.hist(np.clip(error_p_y, -500, 500), bins=200, alpha=0.5, label=f'p_y\nmean:{np.mean(error_p_y):.2f}\nstd:{np.std(error_p_y, ddof=1):.2f}')
plt.hist(np.clip(error_p_z, -500, 500), bins=200, alpha=0.5, label=f'p_z\nmean:{np.mean(error_p_z):.2f}\nstd:{np.std(error_p_z, ddof=1):.2f}')
# plt.hist(error_p_z, bins=100, alpha=0.5)
plt.xlabel('error (gen-est)')
plt.legend()
plt.savefig('./kinematic_graphs/error_nu_1.PNG')
plt.show()

# %%
plt.hist(np.clip(error_E, -500, 500), bins=200, alpha=0.5, label=f'E\nmean:{np.mean(error_E):.2f}\nstd:{np.std(error_E, ddof=1):.2f}')
plt.hist(np.clip(error_p, -500, 500), bins=200, alpha=0.5, label=f'p\nmean:{np.mean(error_p):.2f}\nstd:{np.std(error_p, ddof=1):.2f}')
plt.legend()
plt.xlabel('error (gen-est)')
plt.savefig('./kinematic_graphs/error_nu_2.PNG')
plt.show()

# %%
