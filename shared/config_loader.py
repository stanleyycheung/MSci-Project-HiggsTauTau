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

    def __init__(self, df_inputs, channel='rho_rho'):
        # error in pandas is KeyError
        self.df = df_inputs
        self.channel = channel

    def chooseConfigMap(self, config_num, binary, mode=0):
        """
        Chooses which configurations of inputs to load to NN training
        Modes:
        0: normal configs
        1: with neutrino data
        Returns input features X
        """
        # TODO: fail safe for extra information
        if self.channel == 'rho_rho':
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in mode {mode} (rho-rho channel)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br, ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br, ]
            pi0_1_transformed = np.c_[self.df.pi0_E_1_br, self.df.pi0_px_1_br, self.df.pi0_py_1_br, self.df.pi0_pz_1_br, ]
            pi0_2_transformed = np.c_[self.df.pi0_E_2_br, self.df.pi0_px_2_br, self.df.pi0_py_2_br, self.df.pi0_pz_2_br, ]
            if mode == 0:
                config_map_norho = {
                    1: np.c_[self.df.aco_angle_1],
                    2: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2],
                    # 3: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed],
                    3: np.c_[pi0_1_transformed, pi0_2_transformed, pi_1_transformed, pi_2_transformed],
                    4: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.aco_angle_1],
                    5: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                    6: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                    # adding extra aco angles
                    # 7: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2, self.df.aco_angle_5, self.df.aco_angle_6, self.df.aco_angle_7],
                    # 8: np.c_[self.df.aco_angle_5, self.df.aco_angle_6, self.df.aco_angle_7],
                    # 9: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.aco_angle_1, self.df.aco_angle_5, self.df.aco_angle_6, self.df.aco_angle_7],
                    # 10: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.E_miss],
                    # 11: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi0_2_transformed, self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2, self.df.aco_angle_5, self.df.aco_angle_6, self.df.aco_angle_7, self.df.E_miss],
                }
                return config_map_norho[config_num]
            if mode == 1:
                """
                TO CHANGE
                1: no flag
                2: flag
                3: flag + met
                """
                NR = NeutrinoReconstructor(binary)
                df = None
                if config_num == 1:
                    df = NR.dealWithMissingData(self.df, mode=0)
                elif config_num == 2:
                    df = NR.dealWithMissingData(self.df, mode=1)
                elif config_num == 3:
                    df = NR.dealWithMissingData(self.df, mode=2)
                config_map_neutrino = {
                    1: np.c_[pi0_1_transformed, pi0_2_transformed, pi_1_transformed, pi_2_transformed, df['E_nu_1'], df['E_nu_2'], df['p_t_nu_1'], df['p_t_nu_2'], df['p_z_nu_1'], df['p_z_nu_2'], df['flag']],
                    2: np.c_[pi0_1_transformed, pi0_2_transformed, pi_1_transformed, pi_2_transformed, df['E_nu_1'], df['E_nu_2'], df['p_t_nu_1'], df['p_t_nu_2'], df['p_z_nu_1'], df['p_z_nu_2']],
                    3: np.c_[pi0_1_transformed, pi0_2_transformed, pi_1_transformed, pi_2_transformed, df['E_nu_1'], df['E_nu_2'], df['p_t_nu_1'], df['p_t_nu_2'], df['p_z_nu_1'], df['p_z_nu_2']],

                }
                return config_map_neutrino[config_num]
            else:
                raise Exception('Wrong config mode inputted')
        
        elif self.channel == 'rho_a1':
            # configs 1 and 2 learn, but the others don't
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in mode {mode} (rho-a1 channel)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br, ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br, ]
            pi0_1_transformed = np.c_[self.df.pi0_E_1_br, self.df.pi0_px_1_br, self.df.pi0_py_1_br, self.df.pi0_pz_1_br, ]
            pi2_2_transformed = np.c_[self.df.pi2_E_2_br, self.df.pi2_px_2_br, self.df.pi2_py_2_br, self.df.pi2_pz_2_br, ]
            pi3_2_transformed = np.c_[self.df.pi3_E_2_br, self.df.pi3_px_2_br, self.df.pi3_py_2_br, self.df.pi3_pz_2_br, ]
            if mode == 0:
                config_map_norhoa1 = {
                    1: np.c_[self.df.aco_angle_1],
                    # 1: np.c_[self.df.aco_angle_1, self.df.aco_angle_2],
                    2: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2],
                    3: np.c_[pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, pi_1_transformed, pi_2_transformed],
                    # 4: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, self.df.aco_angle_1],
                    4: np.c_[self.df.aco_angle_1, pi_1_transformed, pi_2_transformed, pi0_1_transformed], # reduced config 4 to exclude all variables not existing in rho-rho
                    5: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                    6: np.c_[pi_1_transformed, pi_2_transformed, pi0_1_transformed, pi2_2_transformed, pi3_2_transformed, self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                }
                return config_map_norhoa1[config_num]
            else:
                raise Exception('Wrong config mode inputted')
                
        elif self.channel == 'a1_a1':
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Loadeded in mode {mode} (a1-a1 channel)~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            pi_1_transformed = np.c_[self.df.pi_E_1_br, self.df.pi_px_1_br, self.df.pi_py_1_br, self.df.pi_pz_1_br, ]
            pi_2_transformed = np.c_[self.df.pi_E_2_br, self.df.pi_px_2_br, self.df.pi_py_2_br, self.df.pi_pz_2_br, ]
            pi2_1_transformed = np.c_[self.df.pi2_E_1_br, self.df.pi2_px_1_br, self.df.pi2_py_1_br, self.df.pi2_pz_1_br, ]
            pi3_1_transformed = np.c_[self.df.pi3_E_1_br, self.df.pi3_px_1_br, self.df.pi3_py_1_br, self.df.pi3_pz_1_br, ]
            pi2_2_transformed = np.c_[self.df.pi2_E_2_br, self.df.pi2_px_2_br, self.df.pi2_py_2_br, self.df.pi2_pz_2_br, ]
            pi3_2_transformed = np.c_[self.df.pi3_E_2_br, self.df.pi3_px_2_br, self.df.pi3_py_2_br, self.df.pi3_pz_2_br, ]
            if mode == 0:
                config_map_noa1a1 = {
                    # 1: np.c_[self.df.aco_angle_1],
                    # 1: np.c_[self.df.aco_angle_1, self.df.aco_angle_2, self.df.aco_angle_3],
                    1: np.c_[self.df.aco_angle_1, self.df.aco_angle_2, self.df.aco_angle_3, self.df.aco_angle_6, self.df.aco_angle_7, self.df.aco_angle_8],
                    2: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2],
                    3: np.c_[pi2_1_transformed, pi3_1_transformed, pi2_2_transformed, pi3_2_transformed, pi_1_transformed, pi_2_transformed],
                    # 4: np.c_[pi_1_transformed, pi_2_transformed, pi2_1_transformed, pi3_1_transformed, pi2_2_transformed, pi3_2_transformed, self.df.aco_angle_1],
                    4: np.c_[self.df.aco_angle_1, pi_1_transformed, pi_2_transformed], # reduced config 4 to exclude all variables not existing in rho-rho
                    5: np.c_[self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                    6: np.c_[pi_1_transformed, pi_2_transformed, pi2_1_transformed, pi3_1_transformed, pi2_2_transformed, pi3_2_transformed, self.df.aco_angle_1, self.df.y_1_1, self.df.y_1_2, self.df.m_1**2, self.df.m_2**2],
                }
                return config_map_noa1a1[config_num]
            else:
                raise Exception('Wrong config mode inputted')
                
        else:
            raise Exception('Channel not understood!')

    def configTrainTestData(self, config_num, binary, mode=0, alt_label=False):
        """
        Loads specific configuration of input NN and splits inputs in test/train set
        """
        try:
            # config_map = self.chooseConfigMap(binary, mode=mode)
            X = self.chooseConfigMap(binary, config_num, mode=mode)
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
    df = pd.read_pickle('input_df/input_rho_rho.pkl')
    binary = False
    CL = ConfigLoader(df)
    X_train, X_test, y_train, y_test = CL.configTrainTestData(3, binary)
    print(X_train, X_test, y_train, y_test)
