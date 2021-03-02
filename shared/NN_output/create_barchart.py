import numpy as np
import matplotlib.pyplot as plt

channel = 'rho_rho'
config_num = 3.6
f = open(f'{channel}_{config_num}_smearing_aucs.txt', 'r')
data = f.read().split('\n')
f.close()

variables = []
degradations = []
optimal_aucs = []
for line in data:
    elements = line.split(',')
    if len(elements) < 3:
        break
    variables.append(elements[0])
    degradations.append(float(elements[1]))
    # optimal_aucs.append(float(elements[3]))
    optimal_aucs.append(float(elements[2]))


plt.figure()
plt.bar(range(len(degradations)), degradations)
x = np.arange(len(variables))
plt.xticks(x, variables, rotation='vertical')
plt.subplots_adjust(bottom=0.15)
plt.ylabel('Degradation of AUC')
<<<<<<< HEAD
plt.title('Degradation of AUCs due to smearing rho-rho channel (config 1.6)')
plt.savefig('degradations_rhorho_1.6.png')
=======
plt.title(f'Degradation of AUCs due to smearing, {channel}, {config_num}')
plt.savefig(f'degradations_{channel}_{config_num}.png')
>>>>>>> 67a6b5fbcafdd947d7b5f3c3ba5f2c36dc520f60
