import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from pylorentz import Momentum4
from data_loader import DataLoader
import os
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

    def __init__(self, variables, channel, df_to_smear, features):
        super().__init__(variables, channel, True)
        self.df_to_smear = df_to_smear
        # particles - list of features columns
        self.features_to_smear = {}
        for f in features:
            if f.startswith('ip') or f.startswith('sv') or f.startswith('pi'):
                if f.startswith('ip'):
                    base_feature = 'ip'
                elif f.startswith('sv'):
                    if self.channel == 'rho_rho':
                        print("SMEARER: NO SV IN RHO_RHO CHANNEL")
                        continue
                    base_feature = 'sv'
                elif f.startswith('pi0'):
                    base_feature = 'pi0'
                elif f.startswith('pi'):
                    base_feature = 'pi'
                if base_feature not in self.features_to_smear:
                    self.features_to_smear[base_feature] = set()
                if '1' in f:
                    if self.channel == 'rho_a1':
                        print("SMEARER: NO SV_1 IN RHO_RHO CHANNEL")
                        continue
                    self.features_to_smear[base_feature].add(base_feature+'_1')
                elif '2' in f:
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


    def createSmearedData(self, from_hdf=False):
        print(f'(Smearer) Loading .root info with using HDF5 as {from_hdf}')
        df_gen_reco = self.readSmearingData(from_hdf=from_hdf)
        print('(Smearer) Cleaning data')
        # df_clean, df_ps_clean, df_sm_clean = self.cleanSmearingData(df_gen_reco)
        df_gen_reco_clean = self.cleanSmearingData(df_gen_reco)
        print('(Smearer) Creating smearing distribution')
        for base_feature in self.features_to_smear:
            self.createSmearedDataForOneBaseFeature(df_gen_reco_clean, self.df_to_smear, base_feature, self.features_to_smear[base_feature])
        return self.df_to_smear

    def createSmearedDataForOneBaseFeature(self, df_gen_reco, df, base_feature, features):
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
                    metx_dist = (df_gen_reco['reco_metx'] - df_gen_reco['metx'])/df_gen_reco['metx']
                    metx_sample = self.inverseTransformSampling(metx_dist, df.shape[0])
                    smeared_metx = df['metx'] + metx_sample
                    smeared_met.append((df['metx'], smeared_metx))
                    df['metx'] = smeared_metx
                elif feature == 'mety':
                    mety_dist = (df_gen_reco['reco_mety'] - df_gen_reco['mety'])/df_gen_reco['mety']
                    mety_sample = self.inverseTransformSampling(mety_dist, df.shape[0])
                    smeared_mety = df['mety'] + mety_sample
                    smeared_met.append((df['mety'], smeared_mety))
                    df['mety'] = smeared_mety
            return np.array(smeared_met)
        elif base_feature == 'ip' or base_feature == 'sv':
            reco_vertex = Momentum4(np.zeros(df.shape[0]), df_gen_reco['reco_'+base_feature+'_x_1'], df_gen_reco['reco_'+base_feature+'_y_1'], df_gen_reco['reco_'+base_feature+'_z_1'])
            gen_vertex = Momentum4(np.zeros(df.shape[0]),  df_gen_reco[base_feature+'_x_1'], df_gen_reco[base_feature+'_y_1'], df_gen_reco[base_feature+'_z_1'])
            eta_dist = reco_vertex.eta - gen_vertex.eta
            phi_dist = reco_vertex.phi - gen_vertex.phi
            p_t_dist = reco_vertex.p_t - gen_vertex.p_t
            eta_dist_sample = self.inverseTransformSampling(eta_dist, df.shape[0])
            phi_dist_sample = self.inverseTransformSampling(phi_dist, df.shape[0])
            p_t_dist_sample = self.inverseTransformSampling(p_t_dist, df.shape[0])
            smeared_vertices = []
            for feature in features:
                label_parts = feature.split('_')
                x_label = label_parts[0]+'_x_'+label_parts[1]
                y_label = label_parts[0]+'_y_'+label_parts[1]
                z_label = label_parts[0]+'_z_'+label_parts[1]
                vertex = Momentum4(np.zeros(df.shape[0]), df[x_label], df[y_label], df[z_label])
                smeared_eta = vertex.eta + eta_dist_sample
                smeared_phi = vertex.phi + phi_dist_sample
                smeared_p_t = vertex.p_t + p_t_dist_sample
                smeared_vertex = Momentum4.e_eta_phi_pt(np.zeros(df.shape[0]), smeared_eta, smeared_phi, smeared_p_t)
                smeared_vertex_list = np.array([(df[x_label], smeared_vertex.p_x), (df[y_label], smeared_vertex.p_y), (df[z_label], smeared_vertex.p_z)])
                df[x_label] = smeared_vertex.p_x
                df[y_label] = smeared_vertex.p_y
                df[z_label] = smeared_vertex.p_z
                smeared_vertices.append(smeared_vertex_list)
            return np.array(smeared_vertex)
        else:
            reco_particle = Momentum4(df_gen_reco['reco_'+base_feature+'_E_1'], df_gen_reco['reco_'+base_feature+'_px_1'], df_gen_reco['reco_'+base_feature+'_py_1'], df_gen_reco['reco_'+base_feature+'_pz_1'])
            gen_particle = Momentum4(df_gen_reco[base_feature+'_E_1'], df_gen_reco[base_feature+'_px_1'], df_gen_reco[base_feature+'_py_1'], df_gen_reco[base_feature+'_pz_1'])
            e_dist = (reco_particle.e - gen_particle.e)/gen_particle.e
            eta_dist = reco_particle.eta - gen_particle.eta
            phi_dist = reco_particle.phi - gen_particle.phi
            p_t_dist = reco_particle.p_t - gen_particle.p_t
            e_dist_sample = self.inverseTransformSampling(e_dist, df.shape[0])
            eta_dist_sample = self.inverseTransformSampling(eta_dist, df.shape[0])
            phi_dist_sample = self.inverseTransformSampling(phi_dist, df.shape[0])
            p_t_dist_sample = self.inverseTransformSampling(p_t_dist, df.shape[0])
            smeared_particles = []
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
                smeared_p_t = particle.p_t + p_t_dist_sample
                smeared_particle = Momentum4.e_eta_phi_pt(smeared_e, smeared_eta, smeared_phi, smeared_p_t)
                smeared_particles_list = np.array([(df[E_label], smeared_particle.e), (df[x_label], smeared_particle.p_x), (df[y_label], smeared_particle.p_y), (df[z_label], smeared_particle.p_z)])
                # print(f'1: {any(np.iscomplex(smeared_particle.e))}')
                # print(f'2: {any(np.iscomplex(smeared_particle.p_x))}')
                # print(f'3: {any(np.iscomplex(smeared_particle.p_y))}')
                # print(f'4: {any(np.iscomplex(smeared_particle.p_z))}')
                df[E_label] = smeared_particle.e
                df[x_label] = smeared_particle.p_x
                df[y_label] = smeared_particle.p_y
                df[z_label] = smeared_particle.p_z
                smeared_particles.append(smeared_particles_list)
            return np.array(smeared_particles)

    def inverseTransformSampling(self, data, n_samples):
        hist, bins = np.histogram(data, bins='scott')
        bin_midpoints = bins[:-1] + np.diff(bins)/2
        cdf = np.cumsum(hist)
        cdf = cdf / cdf[-1]
        values = np.random.rand(n_samples)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = bin_midpoints[value_bins]
        return random_from_cdf

    def plotSmeared(self, from_hdf=True):
        print(f'Loading .root info with using HDF5 as {from_hdf}')
        df_gen_reco = self.readSmearingData(from_hdf=from_hdf)
        print('Cleaning data')
        # df_clean, df_ps_clean, df_sm_clean = self.cleanSmearingData(df_gen_reco)
        df_gen_reco_clean = self.cleanSmearingData(df_gen_reco)
        print('Creating smearing distribution')
        results_all = []
        for base_feature in self.features_to_smear:
            results = self.createSmearedDataForOneBaseFeature(df_gen_reco_clean, self.df_to_smear, base_feature, self.features_to_smear[base_feature])
            results_all.append(results)
        #     for label in results:
        #         plt.figure()
        #         plt.hist(label[0], label='original', alpha=0.5)
        #         plt.hist(label[1], label='smeared', alpha=0.5)
        #         plt.legend()   
        # plt.show()
        # plot first particle graphs
        return results_all
        plt.figure()
        d = pd.DataFrame(np.c_[results[0][0][0][0], results[0][0][0][1]])
        d = d[(d[0]<800) & (d[0]>-0) & (d[1]<800) & (d[1]>-0)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        # plt.plot(np.linspace(0, 800), np.linspace(0, 800), 'r')
        plt.colorbar()
        plt.xlabel('pi_2_E')
        plt.ylabel('smeared_pi_2_E')
        plt.figure()
        d = pd.DataFrame(np.c_[results[0][0][1][0], results[0][0][1][1]])
        d = d[(d[0]<250) & (d[0]>-250) & (d[1]<250) & (d[1]>-250)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('pi_2_px')
        plt.ylabel('smeared_pi_2_px')
        d = pd.DataFrame(np.c_[results[0][0][3][0], results[0][0][3][1]])
        d = d[(d[0]<300) & (d[0]>-300) & (d[1]<300) & (d[1]>-300)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('pi_2_pz')
        plt.ylabel('smeared_pi_2_pz')
        plt.figure()
        d = pd.DataFrame(np.c_[results[1][0][0], results[1][0][1]])
        d = d[(d[0]<800) & (d[0]>-800) & (d[1]<800) & (d[1]>-800)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('met_x')
        plt.ylabel('smeared_met_x')
        d = pd.DataFrame(np.c_[results[1][1][0], results[1][1][1]])
        d = d[(d[0]<800) & (d[0]>-800) & (d[1]<800) & (d[1]>-800)]
        plt.hexbin(d[0], d[1], cmap='viridis', mincnt=None, gridsize=200, bins='log')
        plt.colorbar()
        plt.xlabel('met_y')
        plt.ylabel('smeared_met_y')
        plt.show()
        return results_all

if __name__ == '__main__':
    import config
    variables = config.variables_smearing_rho_rho
    channel = 'rho_rho'
    gen = False
    DL = DataLoader(variables, channel, gen)
    # df = DL.loadRecoData(binary=True, addons=['neutrino', 'met', 'ip', 'sv'])
    # df = DL.readRecoData(from_hdf=True)
    # df_clean, _, _ = DL.cleanRecoData(df)
    df_to_smear = DL.readGenData(from_hdf=True)
    df_to_smear_clean, _, _ = DL.cleanGenData(df_to_smear)
    # particles = ['pi_2', 'pi0_2']
    particles = ['pi_2', 'metx', 'mety', 'ip_1']
    s = Smearer(variables, channel, df_to_smear_clean, particles)
    # print(s.features_to_smear)
    # df = s.createSmearedData(from_hdf=True)
    results = s.plotSmeared(from_hdf=True)
    # print(df.head())
    # print(df.isna().sum())