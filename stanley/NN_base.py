import uproot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class NN_base:
    def __init__(self):
        self.epochs = 0
        self.batch_size = 0
        self.variables = [
            "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand",
            "aco_angle_1", "aco_angle_5", "aco_angle_6", "aco_angle_7", 
            "mva_dm_1","mva_dm_2",
            "tau_decay_mode_1","tau_decay_mode_2",
        #     "ip_x_1", "ip_y_1", "ip_z_1", "ip_x_2", "ip_y_2", "ip_z_2", # ignore impact parameter for now
            "pi_E_1", "pi_px_1", "pi_py_1", "pi_pz_1", 
            "pi_E_2", "pi_px_2", "pi_py_2", "pi_pz_2", 
            "pi0_E_1", "pi0_px_1", "pi0_py_1", "pi0_pz_1",
            "pi0_E_2", "pi0_px_2", "pi0_py_2", "pi0_pz_2", 
            "y_1_1", "y_1_2",
            'met', 'metx', 'mety',
        ]
        self.save_dir = ""
        self.config_str = ""

    def readData(self):
        self.tree_tt = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_tt.root")["ntuple"]
        # self.tree_et = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_et.root")["ntuple"]
        # self.tree_mt = uproot.open("/eos/user/s/stcheung/SWAN_projects/Masters_CP/MVAFILE_AllHiggs_mt.root")["ntuple"]
    
    def cleanData(self):
        self.df = self.tree_tt.pandas.df(self.variables)
        # select only rho-rho events
        self.df_rho = self.df[(self.df['mva_dm_1']==1) & (self.df['mva_dm_2']==1) & (self.df["tau_decay_mode_1"] == 1) & (self.df["tau_decay_mode_2"] == 1)]
        # select ps and sm data
        self.df_rho_ps = self.df_rho[(self.df_rho["rand"]<self.df_rho["wt_cp_ps"]/2)]
        self.df_rho_sm = self.df_rho[(self.df_rho["rand"]<self.df_rho["wt_cp_sm"]/2)]
        # drop unnecessary labels 
        self.df_rho_clean = self.df_rho.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        self.df_rho_ps_clean = self.df_rho_ps.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)
        self.df_rho_sm_clean = self.df_rho_sm.drop(["mva_dm_1","mva_dm_2","tau_decay_mode_1","tau_decay_mode_2", "wt_cp_sm", "wt_cp_ps", "wt_cp_mm", "rand"], axis=1).reset_index(drop=True)

    def plotLoss(self):
        plt.clf()
        # Extract number of run epochs from the training history
        epochs = range(1, len(self.history.history["loss"])+1)
        # Extract loss on training and validation ddataset and plot them together
        plt.figure(figsize=(10,8))
        plt.plot(epochs, self.history.history["loss"], "o-", label="Training")
        plt.plot(epochs, self.history.history["val_loss"], "o-", label="Test")
        plt.xlabel("Epochs"), plt.ylabel("Loss")
        # plt.yscale("log")
        plt.legend()
        plt.savefig(f'./{self.save_dir}/fig/loss_{self.config_str}')
        plt.show()


    def rotation_matrix_from_vectors(self, vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def plot_roc_curve(self, fpr, tpr, auc):
        #  define a function to plot the ROC curves - just makes the roc_curve look nicer than the default
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
        ax.grid()
        ax.text(0.6, 0.3, 'Custom AUC Score: {:.3f}'.format(auc),
                bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'k--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.savefig(f'{self.save_dir}/fig/ROC_curve_{self.config_str}.PNG')