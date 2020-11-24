import numpy as np
from potential_2016 import Potential2016, initNN, runNN, runConfigsNN, runArchitecturesNN, getNN_config


class Potential2016_Neutrino(Potential2016):
    def __init__(self, binary, write_filename='potential_2016_neutrino'):
        super().__init__(binary, write_filename)
        self.save_dir = "potential_2016_neutrino"
        self.load_dir = "potential_2016"
        self.write_dir = 'potential_2016_neutrino'
        self.write_filename = 'potential_2016_neutrino'
        self.file_names = ["pi_1_transformed", "pi_2_transformed", "pi0_1_transformed", "pi0_2_transformed", 
                           "rho_1_transformed", "rho_2_transformed", "aco_angle_1", "y_1_1", "y_1_2", 
                           "m_1", "m_2", "w_a", "w_b", "E_miss", "E_miss_x", "E_miss_y", 
                           "aco_angle_5", "aco_angle_6", "aco_angle_7",
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
            7: np.c_[self.aco_angle_1, self.y_1_1, self.y_1_2, self.E_miss],
            8: np.c_[self.aco_angle_1, self.m_1**2, self.m_2**2, self.y_1_1, self.y_1_2, self.E_miss],
            9: np.c_[self.aco_angle_1, self.pi_1_transformed, self.pi_2_transformed, self.pi0_1_transformed, self.pi0_2_transformed, self.E_miss],
        }
        return [config_map_norho_neutrino]
    
if __name__ == '__main__':
    NN = Potential2016_Neutrino(binary=True, write_filename='potential_2016_neutrino')
    # set up NN
    initNN(NN, read=True)
    # runNN(NN, 3, mode=1)

    # NN_config = getNN_config(2)
    # runArchitecturesNN(NN, NN_config, 3, epochs=50, batch_size=10000, mode=1)
    runConfigsNN(NN, 1, 9, epochs=50, batch_size=10000, mode=1)

    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Finished~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')