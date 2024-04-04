import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_context("paper")
import matplotlib.pyplot as plt

util_BHS = np.load("./BH.npy")

util_FTRL = np.load("./FTRL.npy")
util_FTRL_avg = util_FTRL[:, 0]
util_FTRL_up = util_FTRL[:, 2]
util_FTRL_dn = util_FTRL[:, 1]


util_O_FTRL = np.load("./O_FTRL.npy")
util_O_FTRL_avg = util_O_FTRL[:, 0]
util_O_FTRL_up = util_O_FTRL[:, 2]
util_O_FTRL_dn = util_O_FTRL[:, 1]

util_O_FTRL_0 = np.load("./O_FTRL_0.npy")
util_O_FTRL_0_avg = util_O_FTRL_0[:, 0]
util_O_FTRL_0_up = util_O_FTRL_0[:, 2]
util_O_FTRL_0_dn = util_O_FTRL_0[:, 1]

util_O_FTRL_sin = np.load("./O_FTRL_sin.npy")
util_O_FTRL_sin_avg = util_O_FTRL_sin[:, 0]
util_O_FTRL_sin_up = util_O_FTRL_sin[:, 2]
util_O_FTRL_sin_dn = util_O_FTRL_sin[:, 1]


T = len(util_BHS)

util_BHS /= np.arange(1, T+1)

util_FTRL_avg /= np.arange(1, T+1)
util_FTRL_up /= np.arange(1, T+1)
util_FTRL_dn /= np.arange(1, T+1)

util_O_FTRL_avg /= np.arange(1, T+1)
util_O_FTRL_up /= np.arange(1, T+1)
util_O_FTRL_dn /= np.arange(1, T+1)

util_O_FTRL_0_avg /= np.arange(1, T+1)
util_O_FTRL_0_up /= np.arange(1, T+1)
util_O_FTRL_0_dn /= np.arange(1, T+1)

util_O_FTRL_sin_avg /= np.arange(1, T+1)
util_O_FTRL_sin_up /= np.arange(1, T+1)
util_O_FTRL_sin_dn /= np.arange(1, T+1)

ftrl_regret = util_BHS - util_FTRL_avg
ftrl_regret_up = util_BHS - util_FTRL_up
ftrl_regret_dn = util_BHS - util_FTRL_dn


o_ftrl_regret = util_BHS - util_O_FTRL_avg
o_ftrl_regret_up = util_BHS - util_O_FTRL_up
o_ftrl_regret_dn = util_BHS - util_O_FTRL_dn

o_ftrl_0_regret = util_BHS - util_O_FTRL_0_avg
o_ftrl_0_regret_up = util_BHS - util_O_FTRL_0_up
o_ftrl_0_regret_dn = util_BHS - util_O_FTRL_0_dn

o_ftrl_sin_regret = util_BHS - util_O_FTRL_sin_avg
o_ftrl_sin_regret_up = util_BHS - util_O_FTRL_sin_up
o_ftrl_sin_regret_dn = util_BHS - util_O_FTRL_sin_dn


plt.plot(np.arange(T), ftrl_regret, label=r"FTRL", linewidth=6, color="#50C878", linestyle='--', markevery=2000)
plt.plot(np.arange(T), o_ftrl_0_regret, label=r"OFTRL, $\rho=0$", linewidth=3, color="red", marker='^', markevery=2000, markersize=8)
plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0.75$", linewidth=3, color="green", marker='x', markevery=2000, markersize=10)
plt.plot(np.arange(T), o_ftrl_sin_regret, label=r"OFTRL, $\rho=0.2\sin\left(\frac{t\pi}{10^3}\right)+0.7$", linewidth=3, color="#CD7F32", marker='2', markevery=2000, markersize=9, markeredgecolor='black')
# plt.plot(np.arange(T), ftrl_regret, linewidth=2.5, color="#50C878", linestyle='--', markevery=2000)




plt.legend(fontsize=15)
plt.xlabel("T", fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel(r"$R_T^{(1/2)}/T$", fontsize=17)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylim(-0.01, 1.1)
plt.yticks(fontsize=15)
plt.savefig("./ftrl-us-500.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()

################ STATS ################

##################################
diff = ftrl_regret - o_ftrl_regret
avg_diff = np.mean(diff)
avg_ftrl_regret = np.mean(ftrl_regret)
print(">>",avg_diff/avg_ftrl_regret)
##################################
diff = o_ftrl_0_regret -  ftrl_regret
max_diff = np.max(diff)
max_diff_ind = np.where(diff == max_diff)[0][0]
deter = max_diff/ ftrl_regret[max_diff_ind]
print(deter)





