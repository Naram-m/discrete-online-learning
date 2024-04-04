import numpy as np
from nature import NatureMLDS
# from nature_stationary import CN
from meta_learner import OGDExpertsAgent
import scipy.stats as st
from bh_agent import BHSAgent

T = 4000
trials = 8
util_BHS = []

util_META = []
util_META_bad = []


files = 10379
# files = 1000
max_cache_size = 150
# max_cache_size = 30

nature = NatureMLDS(files=files)
# nature = CN(files = files)
cum_c = np.zeros(files)

bhs_agent = BHSAgent(files=files, max_cache_size=max_cache_size)

meta_agents = [OGDExpertsAgent(files=files, max_cache_size=max_cache_size, rho = 0.5) for i in range(trials)]
meta_bad_agents = [OGDExpertsAgent(files=files, max_cache_size=max_cache_size, rho = 0) for i in range(trials)]

for t in np.arange(1, T + 1):
    if t % 100 == 0:
        print("\nStep: {}\n".format(t))
    c_t = nature.generate_cost()
    # print("Requesting : ", c_t)
    cum_c += c_t
    y_bhs = bhs_agent.step(c_t)
    util_BHS.append(cum_c @ y_bhs)

    meta_good_utils = []
    meta_bad_utils = []

    ################ meta good #################
    for agent_index, meta_good_agent in enumerate(meta_agents) :
        y_ftpl = meta_good_agent.step(c_t)
        meta_good_utils.append(c_t @ y_ftpl)
    avg = np.mean(meta_good_utils)
    a, b = st.t.interval(alpha=0.95, df=len(meta_good_utils) - 1,
                         loc=avg,
                         scale=st.sem(meta_good_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_META.append([avg, a, b])

    ################ meta bad #################
    for agent_index, meta_bad_agent in enumerate(meta_bad_agents) :
        y_ftpl = meta_bad_agent.step(c_t)
        meta_bad_utils.append(c_t @ y_ftpl)
    avg = np.mean(meta_bad_utils)
    a, b = st.t.interval(alpha=0.95, df=len(meta_bad_utils) - 1,
                         loc=avg,
                         scale=st.sem(meta_bad_utils))
    if np.isnan(a) or np.isnan(b):
        a = avg
        b = avg
    util_META_bad.append([avg, a, b])


util_META = np.array(util_META)
util_META = np.cumsum(util_META, 0)

util_META_bad = np.array(util_META_bad)
util_META_bad = np.cumsum(util_META_bad, 0)

with open('./conf_plots/BH.npy', 'wb') as f:
    np.save(f, util_BHS)

with open('./conf_plots/META.npy', 'wb') as f:
    np.save(f, util_META)

with open('./conf_plots/META_BAD.npy', 'wb') as f:
    np.save(f, util_META_bad)