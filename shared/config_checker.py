import os
from data_loader import DataLoader
from alpha_calculator import AlphaCalculator

class ConfigChecker:
    """
    Checks if you have the required files to run a config
    Design should be stateless
    """
    def __init__(self, channel, binary, gen):
        self.channel = channel
        self.binary = binary
        self.gen = gen

    def checkInitialize(self, DL, addons_config, read, from_pickle):
        # parse addons
        result = {}
        if not addons_config:
            addons = []
        else:
            addons = addons_config.keys()
        if read:
            # check loadRecoData
            addons_loaded = ""
            if addons:
                addons_loaded = '_'+'_'.join(addons)
            if not self.gen:
                hdf_file_name = f'{DataLoader.input_df_save_dir_reco}/input_{DL.channel}{addons_loaded}'
            else:
                hdf_file_name = f'{DataLoader.input_df_save_dir_gen}/input_gen_{DL.channel}{addons_loaded}'
            if self.binary:
                hdf_file_name += '_b'
            hdf_file_name += '.h5'
            result[hdf_file_name] = os.path.isfile(hdf_file_name)
        else:
            # check createRecoData
            if from_pickle:
                if not self.gen:
                    result[f"{DL.reco_df_path}_{DL.channel}.h5"] = os.path.isfile(f"{DL.reco_df_path}_{DL.channel}.h5")
                else:
                    result[f"{DL.gen_df_path}_{DL.channel}.h5"] = os.path.isfile(f"{DL.gen_df_path}_{DL.channel}.h5")
            else:
                if not self.gen:
                    result[DL.reco_root_path] = os.path.isfile(DL.reco_root_path)
                else:
                    result[DL.gen_root_path] = os.path.isfile(DL.gen_root_path)
            for addon in addons:
                # check addons
                if addon == 'neutrino':
                    load_alpha = addons_config['neutrino']['load_alpha']
                    termination = addons_config['neutrino']['termination']
                    binary_str = ''
                    if self.binary:
                        binary_str += "_b"
                    if load_alpha:
                        if self.gen:
                            a1 = f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_gen'+binary_str+".npy"
                            a2 = f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_gen'+binary_str+".npy"
                        else:
                            a1 = f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_reco_{termination}'+binary_str+".npy"
                            a2 = f'{AlphaCalculator.alpha_save_dir}/{self.channel}/alpha_1_reco_{termination}'+binary_str+".npy"
                        result[a1] = os.path.isfile(a1)
                        result[a2] = os.path.isfile(a2)
        self.checkResult(result)

    # def checkInitializeGen(self, DL, read, from_pickle):
    #     result = {}
    #     if read:
    #         hdf_file_name = f'{DataLoader.input_df_save_dir_gen}/input_gen_{DL.channel}'
    #         if self.binary:
    #             hdf_file_name += '_b'
    #         hdf_file_name += '.h5'
    #         result[hdf_file_name] = os.path.isfile(hdf_file_name)
    #     else:
    #         # check createRecoData
    #         if from_pickle:
    #             result[f"{DL.gen_df_path}_{DL.channel}.h5"] = os.path.isfile(f"{DL.gen_df_path}_{DL.channel}.h5")
    #         else:
    #             result[DL.gen_root_path] = os.path.isfile(DL.gen_root_path)
    #     self.checkResult(result)

    def checkResult(self, result):
        print('Initialize config checked')
        files_not_exist = []
        for file, loaded in result.items() :
            if loaded is False:
                files_not_exist.append(file)
            print (f'File:{file}: {loaded}')
        if files_not_exist:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~FILES NOT EXISTS~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(*files_not_exist, sep="\n")
            raise SystemExit


    def checkConfig(self, gen, config_num):
        # if gen, no config > 2.6
        # reco rho_rho config 2.8 = 2.7, 2.5 = 2.4 
        if gen:
            if config_num > 2.5 and config_num < 3:
                print(f'Config number {config_num} is not accessible to gen level')
                raise SystemExit
        else:
            if config_num == 2.8:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~WARNING: SAME CONFIG AS 2.7~~~~~~~~~~~~~~~~~~~~~~~~~")
            elif config_num == 2.5:
                print("~~~~~~~~~~~~~~~~~~~~~~~~~WARNING: SAME CONFIG AS 2.4~~~~~~~~~~~~~~~~~~~~~~~~~")
