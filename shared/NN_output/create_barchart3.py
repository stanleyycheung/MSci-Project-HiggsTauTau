import numpy as np
import matplotlib.pyplot as plt

variables = ['rho-a1 1.6', 'rho-a1 3.6', 'a1-a1 1.6', 'a1-a1 3.6']
degradations = [0.1196344987480985, 0.1196344987480985, 0.06712774923987463, 0.07062962002048312]


plt.figure()
plt.bar(range(len(degradations)), degradations)
x = np.arange(len(variables))
plt.xticks(x, variables, rotation='vertical')
plt.subplots_adjust(bottom=0.25)
plt.ylabel('Degradation of AUC')
plt.title('Degradation of AUCs due to smearing after fixing empty dataframes')
plt.savefig('degradations_after_fix.png')
