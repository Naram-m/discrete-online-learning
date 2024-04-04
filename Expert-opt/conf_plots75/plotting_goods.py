import numpy as np
import seaborn as sns
sns.set_theme()
sns.set_context("paper")
import matplotlib.pyplot as plt

util_BHS = np.load("./BH.npy")

util_META = np.load("./META.npy")
util_META_avg = util_META[:, 0]
util_META_up = util_META[:, 2]
util_META_dn = util_META[:, 1]

util_META_bad = np.load("./META_BAD.npy")
util_META_bad_avg = util_META_bad[:, 0]
util_META_bad_up = util_META_bad[:, 2]
util_META_bad_dn = util_META_bad[:, 1]

T = len(util_BHS)

util_BHS /= np.arange(1, T+1)

util_META_avg /= np.arange(1, T+1)
util_META_up /= np.arange(1, T+1)
util_META_dn /= np.arange(1, T+1)

util_META_bad_avg /= np.arange(1, T+1)
util_META_bad_up /= np.arange(1, T+1)
util_META_bad_dn /= np.arange(1, T+1)

meta_regret = util_BHS - util_META_avg
meta_regret_up = util_BHS - util_META_up
meta_regret_dn = util_BHS - util_META_dn

meta_bad_regret = util_BHS - util_META_bad_avg
meta_bad_regret_up = util_BHS - util_META_bad_up
meta_bad_regret_dn = util_BHS - util_META_bad_dn


plt.plot(np.arange(T), meta_regret, label=r"Experts-optimism, $\rho=0.75$", linewidth=3, color="#16A085", marker='o', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), meta_regret_dn, meta_regret_up, color='#16A085', alpha=0.5)

plt.plot(np.arange(T), meta_bad_regret, label=r"Experts-optimism, $\rho=0$", linewidth=3, color="#196F3D", marker='x', markevery=2000, markersize=8)
plt.fill_between(np.arange(T), meta_bad_regret_dn, meta_bad_regret_up, color='#196F3D', alpha=0.5)

# plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0.5$", linewidth=3, color="green", marker='x', markevery=2000, markersize=8)
# plt.fill_between(np.arange(T), o_ftrl_regret_dn, o_ftrl_regret_up, color='green',
#                  alpha=0.5)

# plt.plot(np.arange(T), o_ftrl_regret, label=r"OFTRL, $\rho=0.7$", linewidth=3, color="red", marker='x', markevery=2000,  markersize=10)

# plt.plot(np.arange(T), o_ftpl_0_regret, label=r"OFTPL, $\rho=0$", linewidth=3, color="red", marker='x', markevery=2000,  markersize=10)
# plt.plot(np.arange(T), ftpl_regret, label="FTPL", linewidth=3, color="lightblue", linestyle='dashed')

plt.legend(fontsize=15)
plt.xlabel("T", fontsize=17)
plt.xticks(fontsize=15)
plt.ylabel(r"$R_T/T$", fontsize=17)
plt.yticks(fontsize=15)
plt.savefig("./exps.pdf", bbox_inches = 'tight',pad_inches = 0)
plt.show()
print(meta_bad_regret[-1])