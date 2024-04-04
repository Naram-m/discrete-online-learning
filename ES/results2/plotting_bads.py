import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_context("paper")
import matplotlib.pyplot as plt

util_BHS = np.load("./BH.npy")
# util_FTPL = np.load("./FTPL.npy")

util_O_FTPL_0 = np.load("./O_FTPL_0.npy")
util_O_FTPL_0_avg = util_O_FTPL_0[:, 0]
util_O_FTPL_0_up = util_O_FTPL_0[:, 2]
util_O_FTPL_0_dn = util_O_FTPL_0[:, 1]

util_O_FTRL_0 = np.load("./O_FTRL_0.npy")
util_O_FTRL_0_avg = util_O_FTRL_0[:, 0]
util_O_FTRL_0_up = util_O_FTRL_0[:, 2]
util_O_FTRL_0_dn = util_O_FTRL_0[:, 1]

# util_O_FTPL_0 = np.load("./O_FTPL_0.npy")
# util_O_FTRL = np.load("./O_FTRL.npy")

T = len(util_BHS)

util_BHS /= np.arange(1, T+1)
# util_FTPL /= np.arange(1, T+1)

util_O_FTPL_0_avg /= np.arange(1, T+1)
util_O_FTPL_0_up /= np.arange(1, T+1)
util_O_FTPL_0_dn /= np.arange(1, T+1)

# util_O_FTPL_0 /= np.arange(1, T+1)

# util_O_FTRL /= np.arange(1, T+1)
util_O_FTRL_0_avg /= np.arange(1, T+1)
util_O_FTRL_0_up /= np.arange(1, T+1)
util_O_FTRL_0_dn /= np.arange(1, T+1)


# ftpl_regret = util_BHS - util_FTPL
o_ftpl_regret = util_BHS - util_O_FTPL_0_avg
o_ftpl_regret_up = util_BHS - util_O_FTPL_0_up
o_ftpl_regret_dn = util_BHS - util_O_FTPL_0_dn

# o_ftpl_0_regret = util_BHS - util_O_FTPL_0

# o_ftrl_regret = util_BHS - util_O_FTRL

o_ftrl_regret = util_BHS - util_O_FTRL_0_avg
o_ftrl_regret_up = util_BHS - util_O_FTRL_0_up
o_ftrl_regret_dn = util_BHS - util_O_FTRL_0_dn


plt.plot(np.arange(T), o_ftpl_regret, label=r"OFTPL, $\rho=0$", linewidth=3, color="orange", marker='s', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), o_ftpl_regret_dn, o_ftpl_regret_up, color='orange',
                 alpha=0.5)

plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0$", linewidth=3, color="red", marker='^', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), o_ftrl_regret_dn, o_ftrl_regret_up, color='red',
                 alpha=0.5)

# plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0.7$", linewidth=3, color="red", marker='x', markevery=2000,  markersize=10)

# plt.plot(np.arange(T), o_ftpl_0_regret, label=r"OFTPL, $\rho=0$", linewidth=3, color="red", marker='x', markevery=2000,  markersize=10)
# plt.plot(np.arange(T), ftpl_regret, label="FTPL", linewidth=3, color="lightblue", linestyle='dashed')

plt.legend(fontsize=15)
plt.xlabel("T", fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel(r"$R_T/T$", fontsize=17)
plt.yticks(np.arange(0, 1.1, 0.2))
plt.ylim(-0.12, 1.1)
plt.yticks(fontsize=15)
plt.savefig("./bads.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()
#####################
# to compare to experts
print(o_ftpl_regret[4000])
print(o_ftrl_regret[4000])
print(o_ftpl_regret[-1])
print(o_ftrl_regret[-1])

######################
print("#########################################")
# to calculate the tighter confidence interval
print(o_ftrl_regret_dn[-1] - o_ftrl_regret_up[-1])
print(o_ftpl_regret_dn[-1] - o_ftpl_regret_up[-1])

