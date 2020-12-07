import numpy as np
import matplotlib.pyplot as plt

alpha_1_100 = np.load("alpha_analysis/alpha_1_100.npy", allow_pickle=True)
alpha_1_1000 = np.load("alpha_analysis/alpha_1_1000.npy", allow_pickle=True)


print((alpha_1_100<0).sum())
print((alpha_1_1000<0).sum())
