import os
from data_loader import DataLoader
from alpha_calculator import AlphaCalculator

class ConfigChecker:
    """
    Checks if you have the required files to run a config
    Design should be stateless
    """
    def __init__(self, channel, binary):
        self.binary = binary

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
            pickle_file_name = f'{DataLoader.input_df_save_dir_reco}/input_{DL.channel}{addons_loaded}'
            if self.binary:
                pickle_file_name += '_b'
            pickle_file_name += '.pkl'
            result[pickle_file_name] = os.path.isfile(pickle_file_name)
        else:
            # check createRecoData
            if from_pickle:
                result[f"{DL.reco_df_path}_{DL.channel}.pkl"] = os.path.isfile(f"{DL.reco_df_path}_{DL.channel}.pkl")
            else:
                result[DL.reco_root_path] = os.path.isfile(DL.reco_root_path)
            for addon in addons:
                # check addons
                if addon == 'neutrino':
                    load_alpha = addons_config['neutrino']['load_alpha']
                    termination = addons_config['neutrino']['termination']
                    binary_str = ''
                    if self.binary:
                        binary_str += "_b"
                    if load_alpha:
                        a1 = f'{AlphaCalculator.alpha_save_dir}/alpha_1_{termination}'+binary_str+".npy"
                        a2 = f'{AlphaCalculator.alpha_save_dir}/alpha_2_{termination}'+binary_str+".npy"
                        result[a1] = os.path.isfile(a1)
                        result[a2] = os.path.isfile(a2)
        self.checkResult(result)

    def checkInitializeGen(self, DL, read, from_pickle):
        result = {}
        if read:
            pickle_file_name = f'{DataLoader.input_df_save_dir_gen}/input_gen_{DL.channel}'
            if self.binary:
                pickle_file_name += '_b'
            pickle_file_name += '.pkl'
            result[pickle_file_name] = os.path.isfile(pickle_file_name)
        else:
            # check createRecoData
            if from_pickle:
                result[f"{DL.gen_df_path}_{DL.channel}.pkl"] = os.path.isfile(f"{DL.gen_df_path}_{DL.channel}.pkl")
            else:
                result[DL.gen_root_path] = os.path.isfile(DL.gen_root_path)
        self.checkResult(result)

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