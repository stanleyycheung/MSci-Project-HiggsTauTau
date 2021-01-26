
import numpy as np

# =============================================================================
# auc_single_aco_angle_seeded = [0.5539438243657222, 0.5515396190946614, 0.5533997155569835, \
#                         0.551637309803058, 0.5534583750535915, \
#                             0.5538877065689609, 0.5514907222005615]
#     
# auc_single_aco_angle = [0.5539438243657222, 0.5515396190946614, 0.5533997155569835, \
#                         0.551637309803058, 0.5534583750535915, \
#                             0.5538877065689609, 0.5514907222005615]
#     
# auc_double_aco_angle = [0.550406473450865, 0.5538633507903588, 0.551669042876172, \
#                         0.5539415087621201, 0.5532965588131196, \
#                             0.5507618261568671, 0.5537161423461823]
#     
# single_mean = np.mean(auc_single_aco_angle)
# single_std = np.std(auc_single_aco_angle)
# double_mean = np.mean(auc_double_aco_angle)
# double_std = np.std(auc_double_aco_angle)
# single_seeded_mean = np.mean(auc_single_aco_angle_seeded)
# single_seeded_std = np.std(auc_single_aco_angle_seeded)
# 
# print(single_mean)
# print(single_std)
# print(double_mean)
# print(double_std)
# print(single_seeded_mean)
# print(single_seeded_std)
# =============================================================================


one_aco_angle = [0.534785, 0.534733, 0.534720, 0.534688]
multiple_aco_angles = [0.535200, 0.534960, 0.535237, 0.535103]
aco6 = [0.534438, 0.535712, 0.534300, 0.535745]
one_mean = np.mean(one_aco_angle)
one_std = np.std(one_aco_angle)
multiple_mean = np.mean(multiple_aco_angles)
multiple_std = np.std(multiple_aco_angles)
aco6_mean = np.mean(aco6)
aco6_std = np.std(aco6)

print(one_mean)
print(one_std)
print(multiple_mean)
print(multiple_std)
print(aco6_mean)
print(aco6_std)