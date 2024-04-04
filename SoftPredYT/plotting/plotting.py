'''
BHS and FTPL are taken from P_1
FTRL is taken from R_1
Those (BHS, FTPL (i.e., non opt), FTRL (i.e., non opt) ) were supposed to be the same across all files but ended up being slightly different
due to the seed+sample vs trace saving issue of YTDS.
'''

import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)

util_BHS = np.load("./BH.npy")
T = len(util_BHS)
util_BHS /= np.arange(1, T+1)

util_FTPL = np.load("./FTPL.npy")
util_FTPL_avg = util_FTPL[:, 0]
util_FTPL_avg /= np.arange(1, T+1)
ftpl_regret = util_BHS - util_FTPL_avg
print(ftpl_regret[-1])

util_FTRL = np.load("./FTRL.npy")
util_FTRL_avg = util_FTRL[:, 0]
util_FTRL_avg /= np.arange(1, T+1)
ftrl_regret = util_BHS - util_FTRL_avg
print(ftrl_regret[-1])


util_O_FTPL_0001 = np.load("./O_FTPL_0001.npy")
util_O_FTPL_0001_avg = util_O_FTPL_0001[:, 0]

util_O_FTPL_001 = np.load("./O_FTPL_001.npy")
util_O_FTPL_001_avg = util_O_FTPL_001[:, 0]

util_O_FTPL_01 = np.load("./O_FTPL_01.npy")
util_O_FTPL_01_avg = util_O_FTPL_01[:, 0]

util_O_FTPL_1 = np.load("./O_FTPL_1.npy")
util_O_FTPL_1_avg = util_O_FTPL_1[:, 0]

#########################################

util_O_FTPL_2 = np.load("./O_FTPL_2.npy")
util_O_FTPL_2_avg = util_O_FTPL_2[:, 0]

util_O_FTPL_4 = np.load("./O_FTPL_4.npy")
util_O_FTPL_4_avg = util_O_FTPL_4[:, 0]

util_O_FTPL_8 = np.load("./O_FTPL_8.npy")
util_O_FTPL_8_avg = util_O_FTPL_8[:, 0]

util_O_FTPL_P = np.load("./O_FTPL_P.npy")
util_O_FTPL_P_avg = util_O_FTPL_P[:, 0]

#########################################

util_O_FTPL_0001_avg /= np.arange(1, T+1)
util_O_FTPL_001_avg /= np.arange(1, T+1)
util_O_FTPL_01_avg /= np.arange(1, T+1)
util_O_FTPL_1_avg /= np.arange(1, T+1)

util_O_FTPL_2_avg /= np.arange(1, T+1)
util_O_FTPL_4_avg /= np.arange(1, T+1)
util_O_FTPL_8_avg /= np.arange(1, T+1)
util_O_FTPL_P_avg /= np.arange(1, T+1)


oftpl_0001_regret = util_BHS - util_O_FTPL_0001_avg
oftpl_001_regret = util_BHS - util_O_FTPL_001_avg
oftpl_01_regret = util_BHS - util_O_FTPL_01_avg
oftpl_1_regret = util_BHS - util_O_FTPL_1_avg

oftpl_2_regret = util_BHS - util_O_FTPL_2_avg
oftpl_4_regret = util_BHS - util_O_FTPL_4_avg
oftpl_8_regret = util_BHS - util_O_FTPL_8_avg
oftpl_P_regret = util_BHS - util_O_FTPL_P_avg

##################################################### FTRL ################################################
util_O_FTRL_0001 = np.load("./O_FTRL_0001.npy")
util_O_FTRL_0001_avg = util_O_FTRL_0001[:, 0]

util_O_FTRL_001 = np.load("./O_FTRL_001.npy")
util_O_FTRL_001_avg = util_O_FTRL_001[:, 0]

util_O_FTRL_01 = np.load("./O_FTRL_01.npy")
util_O_FTRL_01_avg = util_O_FTRL_01[:, 0]

util_O_FTRL_1 = np.load("./O_FTRL_1.npy")
util_O_FTRL_1_avg = util_O_FTRL_1[:, 0]

#########################################

util_O_FTRL_2 = np.load("./O_FTRL_2.npy")
util_O_FTRL_2_avg = util_O_FTRL_2[:, 0]

util_O_FTRL_4 = np.load("./O_FTRL_4.npy")
util_O_FTRL_4_avg = util_O_FTRL_4[:, 0]

util_O_FTRL_8 = np.load("./O_FTRL_8.npy")
util_O_FTRL_8_avg = util_O_FTRL_8[:, 0]

util_O_FTRL_P = np.load("./O_FTRL_P.npy")
util_O_FTRL_P_avg = util_O_FTRL_P[:, 0]

#########################################

util_O_FTRL_0001_avg /= np.arange(1, T+1)
util_O_FTRL_001_avg /= np.arange(1, T+1)
util_O_FTRL_01_avg /= np.arange(1, T+1)
util_O_FTRL_1_avg /= np.arange(1, T+1)

util_O_FTRL_2_avg /= np.arange(1, T+1)
util_O_FTRL_4_avg /= np.arange(1, T+1)
util_O_FTRL_8_avg /= np.arange(1, T+1)
util_O_FTRL_P_avg /= np.arange(1, T+1)


oftrl_0001_regret = util_BHS - util_O_FTRL_0001_avg
oftrl_001_regret = util_BHS - util_O_FTRL_001_avg
oftrl_01_regret = util_BHS - util_O_FTRL_01_avg
oftrl_1_regret = util_BHS - util_O_FTRL_1_avg

oftrl_2_regret = util_BHS - util_O_FTRL_2_avg
oftrl_4_regret = util_BHS - util_O_FTRL_4_avg
oftrl_8_regret = util_BHS - util_O_FTRL_8_avg
oftrl_P_regret = util_BHS - util_O_FTRL_P_avg

##################################################### END FTRL ################################################



labels = [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$']
oftpl_means = [oftpl_0001_regret[-1], oftpl_001_regret[-1], oftpl_01_regret[-1], oftpl_1_regret[-1]]
oftpl_means = np.round(oftpl_means, 3)
oftrl_means = [oftrl_0001_regret[-1], oftrl_001_regret[-1], oftrl_01_regret[-1], oftrl_1_regret[-1]]
oftrl_means=np.round(oftrl_means, 3)

labels_2 = ['0.2', '0.4', '0.8', '1.0']
oftpl_means_2 = [oftpl_2_regret[-1], oftpl_4_regret[-1], oftpl_8_regret[-1], oftpl_P_regret[-1]]
oftpl_means_2 = np.round(oftpl_means_2, 3)
oftrl_means_2 = [oftrl_2_regret[-1], oftrl_4_regret[-1], oftrl_8_regret[-1], oftrl_P_regret[-1]]
oftrl_means_2=np.round(oftrl_means_2, 3)


x = np.arange(len(labels))  # the label locations
x2 = np.arange(len(labels_2))  # the label locations

width = 0.3  # the width of the bars

fig, (ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(11,5))


oftpl_rects1 = ax.bar(x - width/2 - 0.04, oftpl_means, width, label='OFTPL', color="blue", alpha=0.9)
oftpl_rects2 = ax2.bar(x2 - width/2 - 0.04, oftpl_means_2, width,  color ="blue", alpha=0.9)

oftrl_rects1 = ax.bar(x + width/2 + 0.04, oftrl_means, width , label='OFTRL', color="green", alpha=0.9)
oftrl_rects2 = ax2.bar(x2 + width/2 + 0.04, oftrl_means_2, width , color ="green", alpha=0.9)

ax.axhline(y = 0.0, color="black", lw=0.5)
ax2.axhline(y = 0.0, color="black", lw=0.5)

ax.axhline(y = ftpl_regret[-1], color="#0096FF",lw=3, linestyle='--',label="FTPL")
ax2.axhline(y = ftpl_regret[-1], color="#0096FF", lw=3,linestyle='--')
ax.axhline(y = ftrl_regret[-1], color="#50C878",lw=3, linestyle=':', label="FTRL")
ax2.axhline(y = ftrl_regret[-1], color="#50C878", lw=3,linestyle=':')


ax.legend(fontsize=12)


# rects2 = ax.bar(x + width/2, women_means, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.


ax.set_ylabel(r'$R_{5k}$', fontsize=17)
# ax.set_xlabel(r'$\zeta$', fontsize=17)
fig.supxlabel(r'$\zeta$', fontsize=17)

ax.set_xticks(x, labels)
ax2.set_xticks(x2, labels_2)

ax.bar_label(oftpl_rects1, padding=3)
ax.bar_label(oftrl_rects1, padding=3)
ax2.bar_label(oftpl_rects2, padding=3)
ax2.bar_label(oftrl_rects2, padding=3)

fig.tight_layout()
plt.savefig("./soft_pred.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()

print("Stats: ")
print("OFTRL",(ftrl_regret[-1] - oftrl_1_regret[-1]) / ftrl_regret[-1]  )
print("OFTPL", (ftpl_regret[-1] - oftpl_1_regret[-1])/ ftpl_regret[-1] )