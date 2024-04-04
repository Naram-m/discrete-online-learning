import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_context("paper")
import matplotlib.pyplot as plt

util_BHS = np.load("./BH.npy")

util_FTPL = np.load("./FTPL.npy")
util_FTPL_avg = util_FTPL[:, 0]
util_FTPL_up = util_FTPL[:, 2]
util_FTPL_dn = util_FTPL[:, 1]


util_O_FTPL = np.load("./O_FTPL.npy")
util_O_FTPL_avg = util_O_FTPL[:, 0]
util_O_FTPL_up = util_O_FTPL[:, 2]
util_O_FTPL_dn = util_O_FTPL[:, 1]

util_O_FTPL_0 = np.load("./O_FTPL_0.npy")
util_O_FTPL_0_avg = util_O_FTPL_0[:, 0]
util_O_FTPL_0_up = util_O_FTPL_0[:, 2]
util_O_FTPL_0_dn = util_O_FTPL_0[:, 1]

util_O_FTPL_sin = np.load("./O_FTPL_sin.npy")
util_O_FTPL_sin_avg = util_O_FTPL_sin[:, 0]
util_O_FTPL_sin_up = util_O_FTPL_sin[:, 2]
util_O_FTPL_sin_dn = util_O_FTPL_sin[:, 1]


T = len(util_BHS)

util_BHS /= np.arange(1, T+1)

util_FTPL_avg /= np.arange(1, T+1)
util_FTPL_up /= np.arange(1, T+1)
util_FTPL_dn /= np.arange(1, T+1)

util_O_FTPL_avg /= np.arange(1, T+1)
util_O_FTPL_up /= np.arange(1, T+1)
util_O_FTPL_dn /= np.arange(1, T+1)

util_O_FTPL_0_avg /= np.arange(1, T+1)
util_O_FTPL_0_up /= np.arange(1, T+1)
util_O_FTPL_0_dn /= np.arange(1, T+1)

util_O_FTPL_sin_avg /= np.arange(1, T+1)
util_O_FTPL_sin_up /= np.arange(1, T+1)
util_O_FTPL_sin_dn /= np.arange(1, T+1)

ftpl_regret = util_BHS - util_FTPL_avg
ftpl_regret_up = util_BHS - util_FTPL_up
ftpl_regret_dn = util_BHS - util_FTPL_dn


o_ftpl_regret = util_BHS - util_O_FTPL_avg
o_ftpl_regret_up = util_BHS - util_O_FTPL_up
o_ftpl_regret_dn = util_BHS - util_O_FTPL_dn

o_ftpl_0_regret = util_BHS - util_O_FTPL_0_avg
o_ftpl_0_regret_up = util_BHS - util_O_FTPL_0_up
o_ftpl_0_regret_dn = util_BHS - util_O_FTPL_0_dn

o_ftpl_sin_regret = util_BHS - util_O_FTPL_sin_avg
o_ftpl_sin_regret_up = util_BHS - util_O_FTPL_sin_up
o_ftpl_sin_regret_dn = util_BHS - util_O_FTPL_sin_dn

plt.plot(np.arange(T), ftpl_regret, label=r"FTPL", linewidth=4, color="#0096FF", linestyle='--', markevery=2000, markersize=8)
plt.plot(np.arange(T), o_ftpl_0_regret, label=r"OFTPL, $\rho=0$", linewidth=3, color="orange", marker='s', markevery=2000, markersize=8)
plt.plot(np.arange(T), o_ftpl_regret, label=r"OFTPL, $\rho=0.75$", linewidth=3, color="blue", marker='o', markevery=2000, markersize=8)
plt.plot(np.arange(T), o_ftpl_sin_regret, label=r"OFTPL, $\rho=0.2\sin\left(\frac{t\pi}{10^3}\right)+0.7$", linewidth=3, color="#E1C16E", marker='p', markevery=2000, markersize=9, markeredgecolor='black')




plt.legend(fontsize=15)
plt.xlabel("T", fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel(r"$R_T/T$", fontsize=17)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.yticks(fontsize=15)
plt.savefig("./ftpl-es.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()
##################################
diff = ftpl_regret -  o_ftpl_regret
avg_diff = np.mean(diff)
avg_ftpl_regret = np.mean(ftpl_regret)
print(">>",avg_diff/avg_ftpl_regret)
##################################
diff = o_ftpl_0_regret -  ftpl_regret
max_diff = np.max(diff)
max_diff_ind = np.where(diff == max_diff)[0][0]
deter = max_diff/ ftpl_regret[max_diff_ind]
print(deter)