
import matplotlib.pyplot as plt
import numpy as np

data_ps = np.random.randn(1000) + 4# * 2 * np.pi
data_sm = np.random.randn(1000) + 3# * 2 * np.pi

myfont = {'fontname':'Arial'}

fig = plt.figure(12)
plt.hist(data_ps, bins=50, alpha=0.5, label='Pseudoscalar')
plt.hist(data_sm, bins=50, alpha=0.5, label='Standard model')
plt.title('Distribution of acoplanarity angle 1', **myfont, fontsize=16)
plt.ylabel('Frequency of events', **myfont, fontsize=14)
plt.xlabel('Acoplanarity angle 1 (rad)', **myfont, fontsize=14)

# fig.patch.set_facecolor('powderblue')
# fig.patch.set_facecolor('lavender')
ax = plt.gca()
# ax.set_facecolor('lightcyan')
# ax.set_facecolor('lavender')
# ax.set_facecolor('aliceblue')
# ax.set_facecolor('ghostwhite')
ax.set_facecolor('#e0e4f4')
# ax.patch.set_alpha(0.1)

legend = plt.legend(prop={'family': 'Arial', 'size': 12})
frame = legend.get_frame()
frame.set_facecolor('#b8c4e4')

plt.xlim(0, 2*np.pi)
plt.savefig("C:/Kristof/Imperial College London Year 4/MSci_project/poster/aco_angle_coloured_test.png", facecolor="#b8c4e4")