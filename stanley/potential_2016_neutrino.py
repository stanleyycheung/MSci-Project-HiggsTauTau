import numpy as np
from potential_2016 import Potential2016, initNN, runNN, runConfigsNN, runArchitecturesNN, getNN_config, Momentum4


class Potential2016_Neutrino(Potential2016):
    def __init__(self, binary, write_filename='potential_2016_neutrino'):
        super().__init__(binary, write_filename)
        self.save_dir = "potential_2016_neutrino"
        self.load_dir = "potential_2016_neutrino"
        self.write_dir = 'potential_2016_neutrino'
        self.write_filename = 'potential_2016_neutrino'
        self.file_names = ["pi_1_transformed", "pi_2_transformed", "pi0_1_transformed", "pi0_2_transformed", 
                           "rho_1_transformed", "rho_2_transformed", "aco_angle_1", "y_1_1", "y_1_2", 
                           "m_1", "m_2", "w_a", "w_b", "E_miss", "E_miss_x", "E_miss_y", 
                           "aco_angle_5", "aco_angle_6", "aco_angle_7",
                           "gen_nu_1_x", "gen_nu_1_y", "gen_nu_1_z", 
                           "gen_nu_1_x", "gen_nu_1_y", "gen_nu_1_z", 
                           "y"]
        if self.binary:
            self.write_filename += '_binary'

        self.variables += ["gen_nu_p_1", "gen_nu_phi_1", "gen_nu_eta_1", #leading neutrino, gen level
                            "gen_nu_p_2", "gen_nu_phi_2", "gen_nu_eta_2" #subleading neutrino, gen level
        ]

    def addConfigs(self):
        config_map_norho_neutrino = {
            1: np.c_[self.aco_angle_1],
            2: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2],
            3: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed],
            4: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1],
            5: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            6: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1 , self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2],
            # adding boosted MET energy
            7: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.E_miss],
            8: np.c_[self.aco_angle_1, self.m_1**2, self.m_2**2, self.y_1_1, self.y_1_2, self.E_miss],
            9: np.c_[self.aco_angle_1, self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.E_miss],
            # adding gen neutrino data
            10: np.c_[self.aco_angle_1, self.E_miss, self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z],
            11: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z],
            12: np.c_[self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1**2, self.m_2**2, self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z],
        }
        return [config_map_norho_neutrino]

    def createExtraData(self, df, boost):
        gen_nu_1_boosted = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_1"])), df["gen_nu_eta_1"], df["gen_nu_phi_1"], df["gen_nu_p_1"]).boost_particle(boost)
        gen_nu_2_boosted = Momentum4.m_eta_phi_p(np.zeros(len(df["gen_nu_phi_2"])), df["gen_nu_eta_2"], df["gen_nu_phi_2"], df["gen_nu_p_2"]).boost_particle(boost)
        self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z = gen_nu_1_boosted.p_x, gen_nu_1_boosted.p_y, gen_nu_1_boosted.p_z
        self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z = gen_nu_2_boosted.p_x, gen_nu_2_boosted.p_y, gen_nu_2_boosted.p_z
        return [self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z]

    def readTrainTestData(self):
        print("Reading train/test files")
        to_load = []
        if not self.binary:
            self.file_names.pop()
        for i in range(len(self.file_names)):
            if self.binary:
                load_name = f'{self.load_dir}/{self.file_names[i]}_b.npy'
            else:
                load_name = f'{self.load_dir}/{self.file_names[i]}.npy'
            to_load.append(np.load(load_name, allow_pickle=True))
        if self.binary:
            self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7, self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z, self.y = to_load
        else:
            self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.rho_1_transformed, self.rho_2_transformed, self.aco_angle_1, self.y_1_1, self.y_1_2, self.m_1, self.m_2, self.w_a, self.w_b, self.E_miss, self.E_miss_x, self.E_miss_y, self.aco_angle_5, self.aco_angle_6, self.aco_angle_7, self.gen_nu_1_x, self.gen_nu_1_y, self.gen_nu_1_z, self.gen_nu_2_x, self.gen_nu_2_y, self.gen_nu_2_z = to_load
        print("Loaded train/test files")

if __name__ == '__main__':
    NN = Potential2016_Neutrino(binary=True, write_filename='potential_2016_neutrino')
    # set up NN
    initNN(NN, read=True)
    # runNN(NN, 3, mode=3)

    # NN_config = getNN_config(2)
    # runArchitecturesNN(NN, NN_config, 3, epochs=50, batch_size=10000, mode=3)
    runConfigsNN(NN, 3, 3, epochs=100, batch_size=100000, mode=3)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')