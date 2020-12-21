
import numpy as np

auc_single_aco_angle_seeded = [0.5539438243657222, 0.5515396190946614, 0.5533997155569835, \
                        0.551637309803058, 0.5534583750535915, \
                            0.5538877065689609, 0.5514907222005615]
    
auc_single_aco_angle = [0.5539438243657222, 0.5515396190946614, 0.5533997155569835, \
                        0.551637309803058, 0.5534583750535915, \
                            0.5538877065689609, 0.5514907222005615]
    
auc_double_aco_angle = [0.550406473450865, 0.5538633507903588, 0.551669042876172, \
                        0.5539415087621201, 0.5532965588131196, \
                            0.5507618261568671, 0.5537161423461823]
    
single_mean = np.mean(auc_single_aco_angle)
single_std = np.std(auc_single_aco_angle)
double_mean = np.mean(auc_double_aco_angle)
double_std = np.std(auc_double_aco_angle)
single_seeded_mean = np.mean(auc_single_aco_angle_seeded)
single_seeded_std = np.std(auc_single_aco_angle_seeded)

print(single_mean)
print(single_std)
print(double_mean)
print(double_std)
print(single_seeded_mean)
print(single_seeded_std)