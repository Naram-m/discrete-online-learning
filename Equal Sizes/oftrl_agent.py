import copy
import numpy as np
import cvxpy as cp
import time

class O_FTRLAgent():
    def __init__(self, files, max_cache_size, acc):
        self.rho = acc
        self.y_dim = files
        self.y = cp.Variable(self.y_dim)
        self.acc_grads = np.zeros(self.y_dim)

        self.max_cache_size = max_cache_size
        self.standard_constraints = []
        self.actions = []
        self.grads = []
        self.acc_sigma = 0
        self.acc_sigma_action = 0
        self.sigmas = []
        ##############
        self.etas = []
        self.acc_grad_square = 0
        self.R = np.sqrt(max_cache_size)  # an upper bound on the l2 norm of x

        ####################################### Constraints ###################################
        self.standard_constraints.extend([self.y >= 0, self.y <= 1])
        self.standard_constraints.extend([cp.sum(self.y) == self.max_cache_size])
        #######################################################################################

        self.non_projected_sol_parameter = cp.Parameter(self.y_dim, nonneg=True)
        self.objective = cp.Minimize(cp.sum_squares(self.y - self.non_projected_sol_parameter))
        self.prob = cp.Problem(self.objective, self.standard_constraints)

    def calc_sigma(self, pred):
        if not self.sigmas:  # no grads yet
            self.etas.append(1)  # for t=0
            self.sigmas.append(1 / self.etas[-1])  # from McMahan \sigma_0  = 1 / \eta_0
        else:
            self.acc_grad_square += np.linalg.norm(self.grads[-1] - pred, 2) ** 2
            self.etas.append(
                np.sqrt(2) * self.R / np.sqrt(self.acc_grad_square))  # this will be the second eta, after 0
            self.sigmas.append(1 / self.etas[-1] - 1 / self.etas[-2])

    def step(self, grad):
        if np.random.rand() <= self.rho:
            pred = grad
        else:
            pred = np.zeros(len(grad))
            pred[np.random.randint(0, len(grad))] = 1

        if not self.actions:  # first action, minimize r_0, returning 0 because it assumed to be ||x||
            yy = np.random.rand(self.y_dim)

        else:
            # 1- calculating x, all involved params should be ready.
            if self.acc_sigma > 0:
                yy = (self.acc_sigma_action + (self.acc_grads + pred)) / self.acc_sigma

            else:  # below is still not a parameter ..
                a = self.acc_grads + pred
                objective = cp.Maximize(a @ self.y)
                prob = cp.Problem(objective, self.standard_constraints)
                result = prob.solve(warm_start=True, ignore_dpp=True)

        # Projection
        if self.acc_sigma > 0 or not self.actions:
            ## testing this
            yy[yy < 0] = 0
            ###############
            self.non_projected_sol_parameter.value = yy
            # start = time.time()
            result = self.prob.solve(warm_start=True, ignore_dpp=True)
            # print("CVX: ", time.time()-start)

        self.actions.append(self.y.value)

        # 2 - preparing params for next round
        self.grads.append(grad)
        self.acc_grads += self.grads[-1]
        self.calc_sigma(pred)
        self.acc_sigma += self.sigmas[-1]
        self.acc_sigma_action += self.sigmas[-1] * self.actions[-1]

        return self.madow_sample_2(self.y.value)

    def madow_sample(self, fractional_var):
        # print("ping")
        integral_var = np.zeros(len(fractional_var))
        cut = np.random.rand()
        for i in np.arange(0, self.max_cache_size):
            for j in np.arange(i, len(fractional_var)):
                shifted_cut = i + cut
                lb = np.sum(fractional_var[0:j])
                ub = lb + fractional_var[j]
                if lb <= shifted_cut < ub:
                    integral_var[j] = 1
        # print("pong")
        return integral_var

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

    def project(unsorted_unprojected, cache_size):
        dim = len(unsorted_unprojected)
        i = np.argsort(unsorted_unprojected)
        sorted_unprojected = unsorted_unprojected[i]
        partial_sums = np.cumsum(sorted_unprojected)
        partial_sums = np.insert(partial_sums, 0, 0)
        sorted_unprojected = np.append(sorted_unprojected, np.inf)
        sorted_unprojected = np.insert(sorted_unprojected, 0, -np.inf)

        break_f = False
        for a in range(0, dim + 1):
            if (cache_size == dim - a) and (sorted_unprojected[a + 1] - sorted_unprojected[a] >= 1):
                b = a
                break
            for b in range(a + 1, dim + 1):
                diff = partial_sums[a] - partial_sums[b]
                gamma = (cache_size + b - dim + diff) / (b - a)
                if (sorted_unprojected[a] + gamma <= 0) and (sorted_unprojected[a + 1] + gamma > 0) and (
                        sorted_unprojected[
                            b] + gamma < 1) and (sorted_unprojected[b + 1] + gamma >= 1):
                    break_f = True
                    break
            if break_f:
                break
        sorted_projected = np.empty(dim)
        # sorted_projected[0:a] = 0
        # sorted_projected[a + 1: b] = sorted_unprojected[a + 1: b] + gamma
        # sorted_projected[b + 1: dim] = 1
        sorted_projected[0: a] = 0  # a will not be included
        sorted_projected[a: b] = sorted_unprojected[
                                 a + 1: b + 1] + gamma  # sorted_unprojected has been appended with the infinities
        sorted_projected[b: dim] = 1

        unsorted_projected = np.empty(dim)
        unsorted_projected[i] = sorted_projected

        return unsorted_projected
