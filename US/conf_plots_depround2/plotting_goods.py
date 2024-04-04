import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_context("paper")
import matplotlib.pyplot as plt

util_BHS = np.load("./BH.npy")
# util_FTPL = np.load("./FTPL.npy")

util_O_FTPL = np.load("./O_FTPL.npy")
util_O_FTPL_avg = util_O_FTPL[:, 0]
util_O_FTPL_up = util_O_FTPL[:, 2]
util_O_FTPL_dn = util_O_FTPL[:, 1]

util_O_FTRL = np.load("./O_FTRL.npy")
util_O_FTRL_avg = util_O_FTRL[:, 0]
util_O_FTRL_up = util_O_FTRL[:, 2]
util_O_FTRL_dn = util_O_FTRL[:, 1]

# util_O_FTPL_0 = np.load("./O_FTPL_0.npy")
# util_O_FTRL = np.load("./O_FTRL.npy")

T = len(util_BHS)

util_BHS /= np.arange(1, T+1)
# util_FTPL /= np.arange(1, T+1)

util_O_FTPL_avg /= np.arange(1, T+1)
util_O_FTPL_up /= np.arange(1, T+1)
util_O_FTPL_dn /= np.arange(1, T+1)

# util_O_FTPL_0 /= np.arange(1, T+1)

# util_O_FTRL /= np.arange(1, T+1)
util_O_FTRL_avg /= np.arange(1, T+1)
util_O_FTRL_up /= np.arange(1, T+1)
util_O_FTRL_dn /= np.arange(1, T+1)


# ftpl_regret = util_BHS - util_FTPL
o_ftpl_regret = util_BHS - util_O_FTPL_avg
o_ftpl_regret_up = util_BHS - util_O_FTPL_up
o_ftpl_regret_dn = util_BHS - util_O_FTPL_dn

# o_ftpl_0_regret = util_BHS - util_O_FTPL_0

# o_ftrl_regret = util_BHS - util_O_FTRL

o_ftrl_regret = util_BHS - util_O_FTRL_avg
o_ftrl_regret_up = util_BHS - util_O_FTRL_up
o_ftrl_regret_dn = util_BHS - util_O_FTRL_dn


plt.plot(np.arange(T), o_ftpl_regret, label=r"OFTPL, $\rho=0.75$", linewidth=3, color="blue", marker='o', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), o_ftpl_regret_dn, o_ftpl_regret_up, color='blue',
                 alpha=0.5)

plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0.75$", linewidth=3, color="green", marker='x', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), o_ftrl_regret_dn, o_ftrl_regret_up, color='green',
                 alpha=0.5)

plt.ylim([-0.02, 1.1])
plt.legend(fontsize=15)
plt.xlabel("T", fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel(r"$R_T^{(1/2)}/T$", fontsize=17)

plt.yticks(fontsize=15)
plt.savefig("./goods-us-500.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()