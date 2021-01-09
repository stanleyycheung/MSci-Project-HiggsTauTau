import scipy.stats as sps
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 14, "figure.figsize": (10,6)})

def profileplot(x, y, xlabel='', ylabel='', bins=100, mode=0):
    means_result = sps.binned_statistic(x, [y, y**2], bins=bins, statistic='mean')
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    # remove NaNs and single count bins
    nan_idx = np.argwhere(np.isnan(means) ).flatten()
    zero_idx = np.argwhere(standard_deviations == 0)
    to_remove = np.union1d(nan_idx, zero_idx)
    means = np.delete(means, to_remove, None)
    bin_centers = np.delete(bin_centers, to_remove, None)
    standard_deviations = np.delete(standard_deviations, to_remove, None)
    count = Counter(means_result.binnumber)
    to_remove_set = set(to_remove)
    N = []
    for i in range(1,bins+1):
        if i-1 in to_remove_set:
            continue
        if i in count:
            N.append(count[i])
    # print(to_remove.shape)
    # print(bin_centers.shape, means.shape)
    yerr = standard_deviations/np.sqrt(N)
    # yerr = standard_deviations
    # fitting
    # print(bin_centers, means, yerr)
    plt.figure()
    plt.errorbar(x=bin_centers, y=means, yerr=yerr, linestyle='none', marker='.', capsize=2)
    if mode == 1:
        fit, cov = np.polyfit(bin_centers, means, 1, w=1/yerr, cov=True)
        p = np.poly1d(fit)
        print(f"Fit params: {fit[0]}, {fit[1]}")
        print(f"Diag of cov: {cov[0][0]} , {cov[1][1]}")
        plt.plot(bin_centers, p(bin_centers))
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    

def profileplot_plain(x, y, xlabel='', ylabel='', bins=100, plot_range=None):
    means_result = sps.binned_statistic(x, [y, y**2], bins=bins, statistic='mean', range=plot_range)
    means, means2 = means_result.statistic
    standard_deviations = np.sqrt(means2 - means**2)
    bin_edges = means_result.bin_edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:])/2.
    plt.figure()
    plt.errorbar(x=bin_centers, y=means, yerr=standard_deviations, linestyle='none', marker='.', capsize=2)
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()