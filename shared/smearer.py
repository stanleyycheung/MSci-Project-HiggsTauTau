import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from pylorentz import Momentum4
from data_loader import DataLoader
import os
import config
# import scipy.interpolate as interpolate

class Smearer(DataLoader):
    """SAVING DF IS NOT SUPPORTED """
    if os.path.exists('/home/hep/shc3117/'):
        print('Running on Imperial HEP LX machines')
        smearing_root_path = '/vols/cms/dw515/Offline/output/SM/master_gen_ntuple_1502/MVAFILE_tt.root'
        smearing_df_path = '/vols/cms/shc3117/df_tt_smearing'
    else:
        smearing_root_path = ''
        smearing_df_path = ''
    # input_df_save_dir_smearing = './input_df_smearing'

    def __init__(self, variables, channel, features):
        super().__init__(variables, channel, True)
        # particles - list of features columns
        self.features_to_smear = {}
        self.pi_mass = {
            'pi': 0.13957,
            'pi2': 0.13957,
            'pi3': 0.13957,
            'pi0': 0.135,
        }
        self.reco_events = {
            'rho_rho': 949753,
            'rho_a1': 507946,
            'a1_a1': 315135,
        }
        for f in features:
            f = str(f)
            if f.startswith('ip') or f.startswith('sv') or f.startswith('pi'):
                if f.startswith('ip'):
                    base_feature = 'ip'
                elif f.startswith('sv'):
                    if self.channel == 'rho_rho':
                        print("SMEARER: NO SV IN RHO_RHO CHANNEL")
                        continue
                    if self.channel == 'rho_a1' and '1' in f:
                        print("SMEARER: NO SV_1 IN RHO_RHO CHANNEL")
                        continue
                    base_feature = 'sv'
                elif f.startswith('pi0'):
                    base_feature = 'pi0'
                elif f.startswith('pi'):
                    base_feature = 'pi'
                if base_feature not in self.features_to_smear:
                    self.features_to_smear[base_feature] = set()
                if f.startswith('pi2'):
                    if '_1' in f:
                        self.features_to_smear[base_feature].add('pi2_1')
                    elif '_2' in f:
                        self.features_to_smear[base_feature].add('pi2_2')
                elif f.startswith('pi3'):
                    if '_1' in f:
                        self.features_to_smear[base_feature].add('pi3_1')
                    elif '_2' in f:
                        self.features_to_smear[base_feature].add('pi3_2')
                elif '_1' in f:
                    self.features_to_smear[base_feature].add(base_feature+'_1')
                elif '_2' in f:
                    self.features_to_smear[base_feature].add(base_feature+'_2')
                else:
                    raise ValueError('Feature direction not found')
            else:
                if 'met' in f:
                    base_feature = 'met'
                else:
                    raise ValueError('Feature not understood')
                if base_feature not in self.features_to_smear:
                    self.features_to_smear[base_feature] = set()
                if 'x' in f:
                    self.features_to_smear[base_feature].add('metx')
                elif 'y' in f:
                    self.features_to_smear[base_feature].add('mety')
                else:
                    raise ValueError('Feature met not understood')
        # print(self.features_to_smear)
    def readSmearingData(self, from_hdf=False):
        if not from_hdf:
            tree_tt = uproot.open(Smearer.smearing_root_path)["ntuple"]
            # new_variables = []
            # keys = [x.decode('utf-8') for x in tree_tt.keys()]
            # for var in self.variables:
            #     if var in keys:
            #         new_variables.append(var)
            # self.variables = new_variables
            df = tree_tt.pandas.df(self.variables)
            df.to_hdf(f"{Smearer.smearing_df_path}_{self.channel}.h5", 'df')
        else:
            df = pd.read_hdf(f"{Smearer.smearing_df_path}_{self.channel}.h5", 'df')
        return df


    def cleanSmearingData(self, df):
        """exactly the same as cleanGenData"""
        if self.channel == 'rho_rho':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 1)]
        elif self.channel == 'rho_a1':
            df_clean = df[(df['dm_1'] == 1) & (df['dm_2'] == 10)]
        elif self.channel == 'a1_a1':
            df_clean = df[(df['dm_1'] == 10) & (df['dm_2'] == 10)]
        else:
            raise ValueError('Incorrect channel inputted')
        df_clean = df_clean.dropna()
        # df_clean = df_clean.loc[~(df_clean == 0).all(axis=1)]
        df_clean = df_clean[(df_clean == -9999.0).sum(1) < 2]
        # df_ps = df_clean[(df_clean["rand"] < df_clean["wt_cp_ps"]/2)]
        # df_sm = df_clean[(df_clean["rand"] < df_clean["wt_cp_sm"]/2)]
        # return df_clean, df_ps, df_sm
        return df_clean

    def selectPSSMFromData(self, df):
        df_ps = df[(df["rand"] < df["wt_cp_ps"]/2)]
        df_sm = df[(df["rand"] < df["wt_cp_sm"]/2)]
        return df_ps, df_sm


    def createSmearedData(self, df_to_smear, from_hdf=False, plot=False, sample=False):
        df_to_smear_copy = df_to_smear.copy()
        print(f'(Smearer) Loading .root info with using HDF5 as {from_hdf}')
        df_gen_reco = self.readSmearingData(from_hdf=from_hdf)
        print('(Smearer) Cleaning data')
        # df_clean, df_ps_clean, df_sm_clean = self.cleanSmearingData(df_gen_reco)
        df_gen_reco_clean = self.cleanSmearingData(df_gen_reco)
        print('(Smearer) Creating smearing distribution')
        for base_feature in self.features_to_smear:
            self.createSmearedDataForOneBaseFeature(df_gen_reco_clean, df_to_smear_copy, base_feature, self.features_to_smear[base_feature], plot=plot)
        print(f'Smeared: {self.features_to_smear}')
        if not sample:
            return df_to_smear_copy
        else:
            return df_to_smear_copy.sample(n=self.reco_events[self.channel], random_state=config.seed_value)

    def createSmearedDataForOneBaseFeature(self, df_gen_reco, df, base_feature, features, plot=False):
        """
        df_gen_reco - df with gen and reco events, to get smearing distribution
        df - df with info to smear - assume channel already selected
        base_feature - type of smearing
        feature - array of specific particles
        replace smeared in df in place
        """
        if base_feature == 'met':
            # energy smearing
            smeared_met = []
            for feature in features:
                if feature == 'metx':
                    metx_dist = df_gen_reco['reco_metx'] - df_gen_reco['metx']
                    metx_sample = self.inverseTransformSampling(metx_dist, df.shape[0])
                    smeared_metx = df['metx'] + metx_sample
                    if plot:
                        plt.figure()
                        d = pd.DataFrame(np.c_[df['metx'], smeared_metx])
                        d = d[(d[0]<500) & (d[0]>-500) & (d[1]<500) & (d[1]>-500)]
                        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                        plt.colorbar()
                        plt.xlabel('met_x')
                        plt.ylabel('smeared_met_x')
                        plt.savefig('./smearing/fig/metx_hexbin.PNG')
                    df['metx'] = smeared_metx
                elif feature == 'mety':
                    mety_dist = df_gen_reco['reco_mety'] - df_gen_reco['mety']
                    mety_sample = self.inverseTransformSampling(mety_dist, df.shape[0])
                    smeared_mety = df['mety'] + mety_sample
                    if plot:
                        plt.figure()
                        d = pd.DataFrame(np.c_[df['mety'], smeared_mety])
                        d = d[(d[0]<500) & (d[0]>-500) & (d[1]<500) & (d[1]>-500)]
                        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                        plt.colorbar()
                        plt.xlabel('met_y')
                        plt.ylabel('smeared_met_y')
                        plt.savefig('./smearing/fig/mety_hexbin.PNG')
                    df['mety'] = smeared_mety
        elif base_feature == 'ip' or base_feature == 'sv':
            # don't smear p_t -> only smear phi and eta
            reco_vertex = Momentum4(np.zeros(df.shape[0]), df_gen_reco['reco_'+base_feature+'_x_1'], df_gen_reco['reco_'+base_feature+'_y_1'], df_gen_reco['reco_'+base_feature+'_z_1'])
            if '1' in base_feature:
                gen_vertex = Momentum4(np.zeros(df.shape[0]),  df_gen_reco[base_feature+'_x_1'], df_gen_reco[base_feature+'_y_1'], df_gen_reco[base_feature+'_z_1'])
            else:
                gen_vertex = Momentum4(np.zeros(df.shape[0]),  df_gen_reco[base_feature+'_x_2'], df_gen_reco[base_feature+'_y_2'], df_gen_reco[base_feature+'_z_2'])

            eta_dist = reco_vertex.eta - gen_vertex.eta
            phi_dist = reco_vertex.phi - gen_vertex.phi
            # p_t_dist = reco_vertex.p_t - gen_vertex.p_t
            eta_dist_sample = self.inverseTransformSampling(eta_dist, df.shape[0])
            phi_dist_sample = self.inverseTransformSampling(phi_dist, df.shape[0])
            # p_t_dist_sample = self.inverseTransformSampling(p_t_dist, df.shape[0])
            for feature in features:
                label_parts = feature.split('_')
                x_label = label_parts[0]+'_x_'+label_parts[1]
                y_label = label_parts[0]+'_y_'+label_parts[1]
                z_label = label_parts[0]+'_z_'+label_parts[1]
                vertex = Momentum4(np.zeros(df.shape[0]), df[x_label], df[y_label], df[z_label])
                smeared_eta = vertex.eta + eta_dist_sample
                smeared_phi = vertex.phi + phi_dist_sample
                # smeared_p_t = vertex.p_t + p_t_dist_sample
                # smeared_vertex = Momentum4.e_eta_phi_pt(np.zeros(df.shape[0]), smeared_eta, smeared_phi, smeared_p_t)
                smeared_vertex = Momentum4.e_eta_phi_pt(np.zeros(df.shape[0]), smeared_eta, smeared_phi, vertex.p_t)
                if plot:
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[x_label], smeared_vertex.p_x])
                    if base_feature == 'ip':
                        d = d[(d[0]<0.05) & (d[0]>-0.05) & (d[1]<0.05) & (d[1]>-0.05)]
                    else:
                        d = d[(d[0]<3) & (d[0]>-3) & (d[1]<3) & (d[1]>-3)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    plt.colorbar()
                    plt.xlabel(f'{x_label}')
                    plt.ylabel(f'smeared_{x_label}')
                    plt.savefig(f'./smearing/fig/{x_label}_hexbin.PNG')
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[y_label], smeared_vertex.p_y])
                    if base_feature == 'ip':
                        d = d[(d[0]<0.05) & (d[0]>-0.05) & (d[1]<0.05) & (d[1]>-0.05)]
                    else:
                        d = d[(d[0]<3) & (d[0]>-3) & (d[1]<3) & (d[1]>-3)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    plt.colorbar()
                    plt.xlabel(f'{y_label}')
                    plt.ylabel(f'smeared_{y_label}')
                    plt.savefig(f'./smearing/fig/{y_label}_hexbin.PNG')
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[z_label], smeared_vertex.p_z])
                    if base_feature == 'ip':
                        d = d[(d[0]<0.05) & (d[0]>-0.05) & (d[1]<0.05) & (d[1]>-0.05)]
                    else:
                        d = d[(d[0]<3) & (d[0]>-3) & (d[1]<3) & (d[1]>-3)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    plt.colorbar()
                    plt.xlabel(f'{z_label}')
                    plt.ylabel(f'smeared_{z_label}')
                    plt.savefig(f'./smearing/fig/{z_label}_hexbin.PNG')
                df[x_label] = smeared_vertex.p_x
                df[y_label] = smeared_vertex.p_y
                df[z_label] = smeared_vertex.p_z
        else:
            # smearing particle
            # 1) smear energy
            # 2) fix mass and recompute p_mag
            # 3) smear phi, eta to get direction
            particle_mass = self.pi_mass[base_feature]
            reco_particle = Momentum4(df_gen_reco['reco_'+base_feature+'_E_1'], df_gen_reco['reco_'+base_feature+'_px_1'], df_gen_reco['reco_'+base_feature+'_py_1'], df_gen_reco['reco_'+base_feature+'_pz_1'])
            gen_particle = Momentum4(df_gen_reco[base_feature+'_E_1'], df_gen_reco[base_feature+'_px_1'], df_gen_reco[base_feature+'_py_1'], df_gen_reco[base_feature+'_pz_1'])
            e_dist = (reco_particle.e - gen_particle.e)/gen_particle.e
            eta_dist = reco_particle.eta - gen_particle.eta
            phi_dist = reco_particle.phi - gen_particle.phi
            e_dist_sample = self.inverseTransformSampling(e_dist, df.shape[0])
            eta_dist_sample = self.inverseTransformSampling(eta_dist, df.shape[0])
            phi_dist_sample = self.inverseTransformSampling(phi_dist, df.shape[0])
            for feature in features:
                label_parts = feature.split('_')
                E_label = label_parts[0]+'_E_'+label_parts[1]
                x_label = label_parts[0]+'_px_'+label_parts[1]
                y_label = label_parts[0]+'_py_'+label_parts[1]
                z_label = label_parts[0]+'_pz_'+label_parts[1]
                particle = Momentum4(df[E_label], df[x_label], df[y_label], df[z_label])
                smeared_e = particle.e*(1+e_dist_sample)
                smeared_eta = particle.eta + eta_dist_sample
                smeared_phi = particle.phi + phi_dist_sample
                smeared_p_mag = np.sqrt(smeared_e**2 - particle_mass**2)
                smeared_p_t = smeared_p_mag/np.cosh(smeared_eta)
                smeared_particle = Momentum4.e_eta_phi_pt(smeared_e, smeared_eta, smeared_phi, smeared_p_t)
                # print(f'1: {any(np.iscomplex(smeared_particle.e))}')
                # print(f'2: {any(np.iscomplex(smeared_particle.p_x))}')
                # print(f'3: {any(np.iscomplex(smeared_particle.p_y))}')
                # print(f'4: {any(np.iscomplex(smeared_particle.p_z))}')
                # print(df[E_label])
                # print(smeared_particle.e)
                if plot:
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[E_label], smeared_e])
                    d = d[(d[0]<800) & (d[0]>-0) & (d[1]<800) & (d[1]>-0)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    # plt.plot(np.linspace(0, 800), np.linspace(0, 800), 'r')
                    plt.colorbar()
                    plt.xlabel(f'{E_label}')
                    plt.ylabel(f'smeared_{E_label}')
                    plt.savefig(f'./smearing/fig/{E_label}_hexbin.PNG')
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[x_label], smeared_particle.p_x])
                    d = d[(d[0]<250) & (d[0]>-250) & (d[1]<250) & (d[1]>-250)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    plt.colorbar()
                    plt.xlabel(f'{x_label}')
                    plt.ylabel(f'smeared_{x_label}')
                    plt.savefig(f'./smearing/fig/{x_label}_hexbin.PNG')
                    plt.figure()
                    d = pd.DataFrame(np.c_[df[z_label], smeared_particle.p_z])
                    d = d[(d[0]<300) & (d[0]>-300) & (d[1]<300) & (d[1]>-300)]
                    plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
                    plt.colorbar()
                    plt.xlabel(f'{z_label}')
                    plt.ylabel(f'smeared_{z_label}')
                    plt.savefig(f'./smearing/fig/{z_label}_hexbin.PNG')
                    plt.figure()
                df[E_label] = smeared_particle.e
                df[x_label] = smeared_particle.p_x
                df[y_label] = smeared_particle.p_y
                df[z_label] = smeared_particle.p_z

    def inverseTransformSampling(self, data, n_samples):
        hist, bins = np.histogram(data, bins='scott')
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(n_samples)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]
        return random_from_cdf

    def plotSmeared(self, df_to_smear,  from_hdf=True):
        print(f'Loading .root info with using HDF5 as {from_hdf}')
        df_gen_reco = self.readSmearingData(from_hdf=from_hdf)
        print('Cleaning data')
        # df_clean, df_ps_clean, df_sm_clean = self.cleanSmearingData(df_gen_reco)
        df_gen_reco_clean = self.cleanSmearingData(df_gen_reco)
        print('Creating smearing distribution')
        results = []
        for base_feature in self.features_to_smear:
            r = self.createSmearedDataForOneBaseFeature(df_gen_reco_clean, df_to_smear, base_feature, self.features_to_smear[base_feature])
            results.append(r)
        #     for label in results:
        #         plt.figure()
        #         plt.hist(label[0], label='original', alpha=0.5)
        #         plt.hist(label[1], label='smeared', alpha=0.5)
        #         plt.legend()   
        # plt.show()
        # plot first particle graphs
        # return results_all

        plt.figure()
        d = pd.DataFrame(np.c_[results[0][0][0][0], results[0][0][0][1]])
        d = d[(d[0]<800) & (d[0]>-0) & (d[1]<800) & (d[1]>-0)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        # plt.plot(np.linspace(0, 800), np.linspace(0, 800), 'r')
        plt.colorbar()
        plt.xlabel('pi_2_E')
        plt.ylabel('smeared_pi_2_E')
        plt.savefig('./smearing/fig/pi_2_E_hexbin.PNG')
        plt.figure()
        d = pd.DataFrame(np.c_[results[0][0][1][0], results[0][0][1][1]])
        d = d[(d[0]<250) & (d[0]>-250) & (d[1]<250) & (d[1]>-250)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('pi_2_px')
        plt.ylabel('smeared_pi_2_px')
        plt.savefig('./smearing/fig/pi_2_px_hexbin.PNG')
        plt.figure()
        d = pd.DataFrame(np.c_[results[0][0][3][0], results[0][0][3][1]])
        d = d[(d[0]<300) & (d[0]>-300) & (d[1]<300) & (d[1]>-300)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('pi_2_pz')
        plt.ylabel('smeared_pi_2_pz')
        plt.savefig('./smearing/fig/pi_2_pz_hexbin.PNG')
        plt.figure()

        d = pd.DataFrame(np.c_[results[1][0][0], results[1][0][1]])
        d = d[(d[0]<800) & (d[0]>-800) & (d[1]<800) & (d[1]>-800)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('met_x')
        plt.ylabel('smeared_met_x')
        plt.savefig('./smearing/fig/metx_hexbin.PNG')
        plt.figure()
        d = pd.DataFrame(np.c_[results[1][1][0], results[1][1][1]])
        d = d[(d[0]<800) & (d[0]>-800) & (d[1]<800) & (d[1]>-800)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('met_y')
        plt.ylabel('smeared_met_y')
        plt.savefig('./smearing/fig/mety_hexbin.PNG')
        plt.show()
        return results

if __name__ == '__main__':
    import config
    variables = config.variables_smearing_rho_a1
    channel = 'rho_a1'
    gen = True
    DL = DataLoader(variables, channel, gen)
    # df = DL.loadRecoData(binary=True, addons=['neutrino', 'met', 'ip', 'sv'])
    # df = DL.readRecoData(from_hdf=True)
    # df_clean, _, _ = DL.cleanRecoData(df)
    df_to_smear = DL.readGenData(from_hdf=True)
    df_to_smear_clean, _, _ = DL.cleanGenData(df_to_smear)
    # particles = ['pi_2', 'metx', 'mety',]
    # particles = ['metx', 'mety']
    particles = ['sv_1', 'sv_2']
    # particles = ['pi_2', 'pi2_2', 'pi3_2', 'pi_1', 'pi2_1', 'pi3_1']
    # particles = ['pi_1']
    s = Smearer(variables, channel, particles)
    # print(s.features_to_smear)
    df = s.createSmearedData(df_to_smear_clean, from_hdf=True, plot=True, sample=True)
    print(df.shape)
    # results = s.plotSmeared(df_to_smear_clean, from_hdf=True)
    # df.to_hdf('./smearing/df_smeared_2.h5', 'df')
    # df_to_smear_clean.to_hdf('./smearing/df_orig_2.h5', 'df')
    # print(df.head())
    # print(df.isna().sum())