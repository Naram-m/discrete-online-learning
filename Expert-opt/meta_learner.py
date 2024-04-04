import copy
import numpy as np
import cvxpy as cp
from optimistic_expert import OptimisticExpert
from robust_expert import RobustExpert


class OGDExpertsAgent():
    def __init__(self, files, max_cache_size, rho):
        self.files = files
        self.w = cp.Variable(2, nonneg=True)

        self.max_cache_size = max_cache_size
        self.standard_constraints = []
        self.actions = []
        self.high_level_grads = []
        ####################################### Constraints ###################################
        # For w:
        self.standard_constraints.extend([cp.sum(self.w) == 1])
        self.standard_constraints.extend([self.w >= 0, self.w <= 1])

        self.non_projected_sol_parameter = cp.Parameter(2, nonneg=True)
        self.objective = cp.Minimize(cp.sum_squares(self.w - self.non_projected_sol_parameter))
        self.prob = cp.Problem(self.objective, self.standard_constraints)

        self.oe = OptimisticExpert(files, max_cache_size, rho=rho)
        self.re = RobustExpert(files, max_cache_size)

    def step(self, grad):
        rea = self.re.step(grad)
        oea = self.oe.step(grad)
        rea[rea < 0] = 0
        # print("REA: ", rea)
        # print("OEA: ", oea)
        # print("The performance of each expert: ", np.round([np.dot(grad, rea), np.dot(grad, oea)], 5))
        if not self.actions:  # first action, minimize r_0, returning 0 because it assumed to be ||x||
            ww = np.array([0.0, 1.0])
        else:
            # ogd_step = np.sqrt(2) * self.R / (1 * np.sqrt(len(self.high_level_grads)))
            ogd_step = 1 / np.sqrt(len(self.high_level_grads))

            ww = self.actions[-1] + ogd_step * self.high_level_grads[-1]

        # Projection
        ww[ww < 0] = 0  # numerical stability
        self.non_projected_sol_parameter.value = ww
        if np.any(ww[ww >= 1e6]):  # numerical stability
            ww[ww >= 1e6] = 1e6
        result = self.prob.solve(warm_start=True, ignore_dpp=True)
        # print("Action is: ", np.round(self.w.value, 5))
        self.actions.append(self.w.value)

        # 2 - preparing params for next round
        self.high_level_grads.append(np.array([np.dot(grad, rea), np.dot(grad, oea)]))

        frac = self.w.value[0] * rea + self.w.value[1] * oea
        # return frac
        frac[frac < 0] = 0
        return self.madow_sample_2(frac)

    def madow_sample_2(self, fractional_var):
        integral_var = np.zeros(len(fractional_var))
        cum_fractional_var = np.cumsum(fractional_var)
        cut = np.random.rand()
        for i in np.arange(0, self.max_cache_size):
            shifted_cut = i + cut
            # print("shifted_cut", shifted_cut)
            # print("cum of last", cum_fractional_var[-1])
            # print(np.where(cum_fractional_var > shifted_cut))
            j = np.where(cum_fractional_var > shifted_cut)[0][0]
            integral_var[j] = 1
        return integral_var