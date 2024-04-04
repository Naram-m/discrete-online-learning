import numpy as np

from bh_agent import BHSAgent

from ftpl_agent import FTPLAgent
from oftpl_agent import O_FTPLAgent

from ftrl_agent import FTRLAgent
from oftrl_agent import O_FTRLAgent

from nature import NatureYTDS
import scipy.stats as st
np.random.seed(42)

T = 5000
trials = 5
util_BHS = []

util_FTPL = []
util_O_FTPL = []
util_O_FTPL_0 = []

util_FTRL = []
util_O_FTRL = []
util_O_FTRL_0 = []

# files = 10379
files = 10000
max_cache_size = 150

nature = NatureYTDS(files=files)

cum_c = np.zeros(files)

bhs_agent = BHSAgent(files=files, max_cache_size=max_cache_size)

ftpl_agents = [FTPLAgent(files=files, max_cache_size=max_cache_size) for i in range(trials)]
o_ftpl_agents = [O_FTPLAgent(files=files, max_cache_size=max_cache_size, acc=1.0) for i in range(trials)]

ftrl_agents = [FTRLAgent(files=files, max_cache_size=max_cache_size) for i in range(trials)]
o_ftrl_agents = [O_FTRLAgent(files=files, max_cache_size=max_cache_size, acc=1.0) for i in range(trials)]

for t in np.arange(1, T + 1):
    if t % 100 == 0:
        print("\nStep: {}\n".format(t))
    c_t = nature.generate_cost()
    cum_c += c_t
    y_bhs = bhs_agent.step(c_t)
    util_BHS.append(cum_c @ y_bhs)

    agents_utils = []

    oftpl_agents_utils = []
    oftpl_agents_0_utils = []

    ftrl_agents_utils = []

    oftrl_agents_utils = []
    oftrl_agents_0_utils = []

    ################ Regular FTPL #################
    for agent_index, ftpl_agent in enumerate(ftpl_agents) :
        y_ftpl = ftpl_agent.step(c_t)
        agents_utils.append(c_t @ y_ftpl)
    avg = np.mean(agents_utils)
    a, b = st.t.interval(alpha=0.95, df=len(agents_utils) - 1,
                         loc=avg,
                         scale=st.sem(agents_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_FTPL.append([avg, a, b])

    ################ FTPL good #################
    for agent_index, o_ftpl_agent in enumerate(o_ftpl_agents):
        y_o_ftpl = o_ftpl_agent.step(c_t)
        oftpl_agents_utils.append(c_t @ y_o_ftpl)
    avg = np.mean(oftpl_agents_utils)
    a, b = st.t.interval(alpha=0.95, df=len(oftpl_agents_utils) - 1,
                         loc=avg,
                         scale=st.sem(oftpl_agents_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_O_FTPL.append([avg, a, b])

    ################ Regular FTRL #################
    for agent_index, ftrl_agent in enumerate(ftrl_agents) :
        y_ftrl = ftrl_agent.step(c_t)
        ftrl_agents_utils.append(c_t @ y_ftrl)
    avg = np.mean(ftrl_agents_utils)
    a, b = st.t.interval(alpha=0.95, df=len(ftrl_agents_utils) - 1,
                         loc=avg,
                         scale=st.sem(ftrl_agents_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_FTRL.append([avg, a, b])

    ################ FTRL good #################
    for agent_index, o_ftrl_agent in enumerate(o_ftrl_agents) :
        y_o_ftrl = o_ftrl_agent.step(c_t)
        oftrl_agents_utils.append(c_t @ y_o_ftrl)
    avg = np.mean(oftrl_agents_utils)
    a, b = st.t.interval(alpha=0.95, df=len(oftrl_agents_utils) - 1,
                         loc=avg,
                         scale=st.sem(oftrl_agents_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_O_FTRL.append([avg, a, b])


# util_FTPL = np.cumsum(util_FTPL)
util_FTPL = np.array(util_FTPL)
util_FTPL = np.cumsum(util_FTPL, 0)

util_O_FTPL = np.array(util_O_FTPL)
util_O_FTPL = np.cumsum(util_O_FTPL, 0)

# util_O_FTPL_0 = np.cumsum(util_O_FTPL_0)
util_O_FTPL_0 = np.array(util_O_FTPL_0)
util_O_FTPL_0 = np.cumsum(util_O_FTPL_0, 0)

util_O_FTRL = np.array(util_O_FTRL)
util_O_FTRL = np.cumsum(util_O_FTRL, 0)

# util_O_FTRL_0 = np.cumsum(util_O_FTRL_0)
util_O_FTRL_0 = np.array(util_O_FTRL_0)
util_O_FTRL_0 = np.cumsum(util_O_FTRL_0, 0)

util_FTRL = np.array(util_FTRL)
util_FTRL = np.cumsum(util_FTRL, 0)


with open('PP/BH.npy', 'wb') as f:
    np.save(f, util_BHS)

with open('PP/FTPL.npy', 'wb') as f:
    np.save(f, util_FTPL)

with open('PP/O_FTPL_P.npy', 'wb') as f:
    np.save(f, util_O_FTPL)


with open('RP/BH.npy', 'wb') as f:
    np.save(f, util_BHS)

with open('RP/FTRL.npy', 'wb') as f:
    np.save(f, util_FTRL)

with open('RP/O_FTRL_P.npy', 'wb') as f:
    np.save(f, util_O_FTRL)