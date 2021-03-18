import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import metrics

channel = 'rho_a1'

fpr_files = [x for x in os.listdir() if x.startswith('fpr')]
tpr_files = [x for x in os.listdir() if x.startswith('tpr')]
fpr_configs = [float(x[26:29]) for x in fpr_files]
tpr_configs = [float(x[26:29]) for x in tpr_files]
fpr_dict = {}
tpr_dict = {}
for i in range(len(fpr_files)):
    f = open(fpr_files[i], 'r')
    data_fpr = [float(x) for x in f.read().split(' ') if x!='' and x!=' ']
    f.close()
    fpr_dict[fpr_configs[i]] = data_fpr
    f = open(tpr_files[i], 'r')
    data_tpr = [float(x) for x in f.read().split(' ') if x!='' and x!=' ']
    tpr_dict[tpr_configs[i]] = data_tpr
#print(fpr_dict)
#print(tpr_dict)

fig, ax = plt.subplots()
for key in sorted(fpr_dict.keys()):
#for key in fpr_dict.keys():
    auc = metrics.auc(fpr_dict[key], tpr_dict[key])
    ax.plot(fpr_dict[key], tpr_dict[key], label='config '+str(key)+' - auc = {:.3f}'.format(auc), alpha=0.5)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax.grid()
    #ax.text(0.6, 0.3, 'Custom AUC Score: {:.3f}'.format(auc),
    #        bbox=dict(boxstyle='square,pad=0.3', fc='white', ec='k'))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    print(key)
plt.legend()
plt.title(f'Multiple ROC curves {channel}')
plt.savefig(f'multiple_roc_{channel}.png')