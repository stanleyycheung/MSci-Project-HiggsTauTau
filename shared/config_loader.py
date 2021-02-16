from multiprocessing import Value
import numpy as np
from sklearn.model_selection import train_test_split
from neutrino_reconstructor import NeutrinoReconstructor

class ConfigLoader:
    """
    ConfigLoader class
    Functions:
    - Loads specific configurations given a df of possible NN inputs
    - Splits the input events into testing and training set
    TODO:
    - Support more maps
    """

    def __init__(self, df_inputs, channel: str, gen: bool):
        # error in pandas is KeyError
        self.df = df_inputs
        self.channel = channel
        self.gen = gen

    def chooseConfigMap(self, config_num: int, binary: bool):
        """
        Chooses which configurations of inputs to load to NN training
        Modes:
        0: normal configs
        1: with neutrino data
        Returns input features X

        Config number naming:
        1 - normal configurations
        2 - with information already in the .root file

        """
        # TODO: fail safe for extra information
        if self.channel == 'rho_rho':
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in {config_num} in rho-rho channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br ]
            pi0_1_transformed = np.c_[self.df.pi0_E_1_br, self.df.pi0_px_1_br, self.df.pi0_py_1_br, self.df.pi0_pz_1_br ]
            pi0_2_transformed = np.c_[self.df.pi0_E_2_br, self.df.pi0_px_2_br, self.df.pi0_py_2_br, self.df.pi0_pz_2_br ]
            four_vectors = np.c_[pi0_1_transformed, pi0_2_transformed, pi_1_transformed, pi_2_transformed]
            aco_angles_calc = np.c_[self.df.aco_angle_1_calc]
            y_s = np.c_[self.df.y_rho_1, self.df.y_rho_2]            
            m_s = np.c_[self.df.m_rho_1**2, self.df.m_rho_2**2]
        elif self.channel == 'rho_a1':
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in {config_num} in rho-a1 channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br ]
            pi0_1_transformed = np.c_[self.df.pi0_E_1_br, self.df.pi0_px_1_br, self.df.pi0_py_1_br, self.df.pi0_pz_1_br ]
            pi2_2_transformed = np.c_[self.df.pi2_E_2_br, self.df.pi2_px_2_br, self.df.pi2_py_2_br, self.df.pi2_pz_2_br ]
            pi3_2_transformed = np.c_[self.df.pi3_E_2_br, self.df.pi3_px_2_br, self.df.pi3_py_2_br, self.df.pi3_pz_2_br ]
            four_vectors = np.c_[pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, pi_1_transformed, pi_2_transformed]
            aco_angles_calc = np.c_[self.df.aco_angle_1_calc, self.df.aco_angle_2_calc, self.df.aco_angle_3_calc, self.df.aco_angle_4_calc]
            y_s = np.c_[self.df.y_rho_1, self.df.y_rho0_2, self.df.y_rho02_2, self.df.y_a1_2, self.df.y_a12_2]
            m_s = np.c_[self.df.m_rho_1**2, self.df.m_rho0_2**2, self.df.m_rho02_2**2, self.df.m_a1_2**2]
        elif self.channel == 'a1_a1':
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in {config_num} in a1-a1 channel~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br ]
            pi2_1_transformed = np.c_[self.df.pi2_E_1_br, self.df.pi2_px_1_br, self.df.pi2_py_1_br, self.df.pi2_pz_1_br ]
            pi3_1_transformed = np.c_[self.df.pi3_E_1_br, self.df.pi3_px_1_br, self.df.pi3_py_1_br, self.df.pi3_pz_1_br ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br ]
            pi2_2_transformed = np.c_[self.df.pi2_E_2_br, self.df.pi2_px_2_br, self.df.pi2_py_2_br, self.df.pi2_pz_2_br ]
            pi3_2_transformed = np.c_[self.df.pi3_E_2_br, self.df.pi3_px_2_br, self.df.pi3_py_2_br, self.df.pi3_pz_2_br ]
            four_vectors = np.c_[pi2_1_transformed, pi3_1_transformed, pi2_2_transformed, pi3_2_transformed, pi_1_transformed, pi_2_transformed]
            aco_angles_calc = np.c_[self.df.aco_angle_1_calc, self.df.aco_angle_2_calc, self.df.aco_angle_3_calc, self.df.aco_angle_4_calc, self.df.aco_angle_5_calc, self.df.aco_angle_6_calc, self.df.aco_angle_7_calc, self.df.aco_angle_8_calc, self.df.aco_angle_9_calc, self.df.aco_angle_10_calc, self.df.aco_angle_11_calc,self.df.aco_angle_12_calc, self.df.aco_angle_13_calc, self.df.aco_angle_14_calc, self.df.aco_angle_15_calc, self.df.aco_angle_16_calc]
            y_s = np.c_[self.df.y_rho0_1, self.df.y_rho02_1, self.df.y_rho_2, self.df.y_rho0_2, self.df.y_a1_1, self.df.y_a12_1, self.df.y_a1_2, self.df.y_a12_2]
            m_s = np.c_[self.df.m_rho0_2**2, self.df.m_rho02_2**2, self.df.m_rho0_2**2, self.df.m_rho02_2**2, self.df.m_a1_1**2, self.df.m_a1_1**2]
        else:
            raise Exception('Channel not understood!')

        if config_num < 2:
            if config_num == 1.1:
                return np.c_[aco_angles_calc]
            elif config_num == 1.2:
                return np.c_[aco_angles_calc, y_s]
            elif config_num == 1.3:
                return np.c_[four_vectors]
            elif config_num == 1.4:
                return np.c_[aco_angles_calc, four_vectors]
            elif config_num == 1.5:
                return np.c_[aco_angles_calc, y_s, m_s]
            elif config_num == 1.6:
                return np.c_[four_vectors, aco_angles_calc, y_s, m_s]
            else:
                return ValueError('Subconfig in config 1 not understood')
        elif config_num < 3:
            if config_num == 2.1:
                # met configurations - following ppt before
                return np.c_[self.df.metx_b, self.df.mety_b, aco_angles_calc, y_s]
            elif config_num == 2.2:
                return np.c_[self.df.metx_b, self.df.mety_b, aco_angles_calc, m_s]
            elif config_num == 2.3:
                return np.c_[self.df.metx_b, self.df.mety_b, aco_angles_calc, four_vectors]
            elif config_num == 2.4:
                # met + config 1.6
                return np.c_[four_vectors, aco_angles_calc, y_s, m_s, self.df.metx_b, self.df.mety_b]
            elif config_num == 2.5:
                # add sv + config 2.4
                base = np.c_[four_vectors, aco_angles_calc, y_s, m_s, self.df.metx_b, self.df.mety_b]
                if self.channel == 'a1_a1' or self.gen:
                    sv_1_transformed = np.c_[self.df.sv_x_1_br, self.df.sv_y_1_br, self.df.sv_z_1_br]
                    sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                    return np.c_[base, sv_1_transformed, sv_2_transformed, self.df.pv_angle]
                elif self.channel == 'rho_a1':
                    sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                    return np.c_[base, sv_2_transformed]
                else:
                    print('SAME AS CONFIG 2.4')
                    return np.c_[base]
            elif config_num > 2.5:
                # not accessible to gen level
                if self.gen:
                    raise ValueError('Configs >2.5 are not accessible to gen level')
                ip_1_transformed = np.c_[self.df.ip_x_1_br, self.df.ip_y_1_br, self.df.ip_z_1_br]
                ip_2_transformed = np.c_[self.df.ip_x_2_br, self.df.ip_y_2_br, self.df.ip_z_2_br]
                if config_num == 2.6:
                    # add ips
                    return np.c_[ip_1_transformed, ip_2_transformed, four_vectors, aco_angles_calc, y_s, m_s]
                elif config_num == 2.7:
                    # met + ip + config 1.6
                    return np.c_[ip_1_transformed, ip_2_transformed, four_vectors, aco_angles_calc, y_s, m_s, self.df.metx_b, self.df.mety_b]
                elif config_num == 2.8:
                    # adding all other additional info in .root file - no aco angles or y
                    base = np.c_[ip_1_transformed, ip_2_transformed, four_vectors, aco_angles_calc, y_s, m_s, self.df.metx_b, self.df.mety_b]
                    if self.channel == 'rho_rho':
                        print('SAME AS CONFIG 2.7')
                        return np.c_[base]
                    elif self.channel == 'rho_a1':
                        sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                        return np.c_[base, sv_2_transformed]
                    else:
                        sv_1_transformed = np.c_[self.df.sv_x_1_br, self.df.sv_y_1_br, self.df.sv_z_1_br]
                        sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                        return np.c_[base, sv_1_transformed, sv_2_transformed, self.df.pv_angle]
                elif config_num == 2.9:
                    # adding all other additional info in .root file
                    base = np.c_[ip_1_transformed, ip_2_transformed, four_vectors, aco_angles_calc, y_s, m_s, self.df.metx_b, self.df.mety_b]
                    if self.channel == 'rho_rho':
                        return np.c_[base, self.df.aco_angles_1, self.df.aco_angles_5, self.df.aco_angles_6, self.df.aco_angles_7, self.df.y_1_1, self.df.y_1_2]
                    elif self.channel == 'rho_a1':
                        sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                        return np.c_[base, self.df.aco_angles_1, self.df.aco_angles_2, self.df.aco_angles_3, self.df.aco_angles_4, self.df.y_1_1, self.df.y_1_2, self.df.y_2_2, self.df.y_3_2, self.df.y_4_2, sv_2_transformed]
                    else:
                        sv_1_transformed = np.c_[self.df.sv_x_1_br, self.df.sv_y_1_br, self.df.sv_z_1_br]
                        sv_2_transformed = np.c_[self.df.sv_x_2_br, self.df.sv_y_2_br, self.df.sv_z_2_br]
                        return np.c_[base, self.df.aco_angles_1, self.df.aco_angles_2, self.df.aco_angles_3, self.df.aco_angles_4, self.df.y_1_1, self.df.y_1_2, self.df.y_2_2, self.df.y_3_2, self.df.y_4_2, sv_1_transformed, sv_2_transformed, self.df.pv_angle]
            else:
                return ValueError('Subconfig in config 2 not understood')

        elif config_num < 4:
            NR = NeutrinoReconstructor(binary)
            pass
        else:
            return ValueError('Config number is not understood')

    def configTrainTestData(self, config_num: int, binary: bool, alt_label: str=False):
        """
        Loads specific configuration of input NN and splits inputs in test/train set
        """
        try:
            # config_map = self.chooseConfigMap(binary, mode=mode)
            X = self.chooseConfigMap(config_num, binary)
        except KeyError as e:
            raise ValueError(f'Wrong config input : {e}')
        # X = config_map[config_num]
        if binary:
            y = self.df['y']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456, stratify=y)
        else:
            if alt_label:
                y = (self.df.w_a > self.df.w_b).astype(int)
            else:
                y = (self.df.w_a/(self.df.w_a+self.df.w_b))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123456,)
        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    import pandas as pd
    from data_loader import DataLoader
    import config
    addons_config = {}
    read = True
    from_pickle = True
    binary = True
    DL = DataLoader(config.variables_rho_rho, 'rho_rho', gen=False)
    df = DL.loadRecoData(binary)

    CL = ConfigLoader(df, 'rho_rho')
    X_train, X_test, y_train, y_test = CL.configTrainTestData(6, binary)
    print(X_train.shape, X_test.shape)
    print(y_train, y_test)
