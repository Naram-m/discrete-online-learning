import numpy as np

from bh_agent import BHSAgent

from oftpl_sin import O_FTPLAgent_SIN
#
from oftrl_sin import O_FTRLAgent_SIN

from nature import NatureMLDS
import scipy.stats as st

T = 10000
trials = 8
util_BHS = []

util_OFTPL_SIN = []
util_OFTRL_SIN = []
files = 10379

# max_cache_size = 150
max_cache_size = 500


nature = NatureMLDS(files=files)

cum_c = np.zeros(files)

bhs_agent = BHSAgent(files=files, max_cache_size=max_cache_size)


o_ftpl_agents_sin = [O_FTPLAgent_SIN(files=files, max_cache_size=max_cache_size, acc='b') for i in range(trials)]
o_ftrl_agents_sin = [O_FTRLAgent_SIN(files=files, max_cache_size=max_cache_size, acc='a') for i in range(trials)]

for t in np.arange(1, T + 1):
    if t % 100 == 0:
        print("\nStep: {}\n".format(t))
    c_t = nature.generate_cost()
    cum_c += c_t
    y_bhs = bhs_agent.step(c_t)
    util_BHS.append(cum_c @ y_bhs)

    oftpl_agents_sin_utils = []
    oftrl_agents_sin_utils = []

    ################ Regular FTPL #################
    for agent_index, ftpl_agent in enumerate(o_ftpl_agents_sin) :
        y_ftpl_sin = ftpl_agent.step(c_t)
        oftpl_agents_sin_utils.append(c_t @ y_ftpl_sin)
    avg = np.mean(oftpl_agents_sin_utils)
    a, b = st.t.interval(alpha=0.95, df=len(oftpl_agents_sin_utils) - 1,
                         loc=avg,
                         scale=st.sem(oftpl_agents_sin_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_OFTPL_SIN.append([avg, a, b])

    ####################################################
    ################ Regular FTRL #################
    for agent_index, ftrl_agent in enumerate(o_ftpl_agents_sin) :
        y_ftrl_sin = ftrl_agent.step(c_t)
        oftrl_agents_sin_utils.append(c_t @ y_ftrl_sin)
    avg = np.mean(oftrl_agents_sin_utils)
    a, b = st.t.interval(alpha=0.95, df=len(oftrl_agents_sin_utils) - 1,
                         loc=avg,
                         scale=st.sem(oftrl_agents_sin_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_OFTRL_SIN.append([avg, a, b])

    ################ FTRL good #################

util_OFTPL_SIN = np.array(util_OFTPL_SIN)
util_OFTPL_SIN = np.cumsum(util_OFTPL_SIN, 0)

#
util_OFTRL_SIN = np.array(util_OFTRL_SIN)
util_OFTRL_SIN = np.cumsum(util_OFTRL_SIN, 0)

with open('./conf_plots_depround2/O_FTPL_SIN.npy', 'wb') as f:
    np.save(f, util_OFTPL_SIN)

with open('./conf_plots_depround2/O_FTRL_SIN.npy', 'wb') as f:
    np.save(f, util_OFTRL_SIN)
