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
        self.R = np.sqrt(max_cache_size)  # an upper bound on the l2 norm of x (20 Jan: not diameter?)

        ####################################### Constraints ###################################
        self.weights = np.load("./weights.npy")
        self.standard_constraints.extend([self.y >= 0, self.y <= 1])
        # self.standard_constraints.extend([cp.sum(self.y) == self.max_cache_size])
        self.standard_constraints.extend([self.weights @ self.y <= self.max_cache_size])
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

        # 1- self.y.value to almost integral:
        # almost_integral = self.dep_round(self.y.value)
        almost_integral = self.dep_round_2(self.y.value)
        # 2- almost_integral to randomized
        frac_index = np.where(np.logical_and(almost_integral < 1, almost_integral > 0))[0]
        rand_sol_cand_1 = copy.copy(almost_integral)
        rand_sol_cand_2 = np.zeros(len(almost_integral))
        if np.size(frac_index) >= 1:
            # rand_sol_cand_1[frac_index] = 0
            # rand_sol_cand_2[frac_index] = 1
            rand_sol_cand_1[frac_index[0]] = 0
            rand_sol_cand_2[frac_index[0]] = 1

        if np.random.random() <= 0.5:
            random_sol = rand_sol_cand_1
        else:
            random_sol = rand_sol_cand_2
        return random_sol

    def dep_round(self, frac):
        frac_copy = copy.copy(frac)
        fracs_indicies = np.where(np.logical_and(frac_copy > 0, frac_copy < 1))[0]
        while np.size(fracs_indicies) >= 2:  # there exist at least one fractional element
            i = fracs_indicies[0]
            j = fracs_indicies[1]
            beta_1 = frac_copy[i]
            beta_2 = frac_copy[j]
            if 0 <= beta_1 + beta_2 <= 1:
                sample = np.random.rand()
                if sample <= beta_2 / (beta_1 + beta_2):
                    frac_copy[i] = 0
                else:
                    frac_copy[j] = 0
            elif 1 <= beta_1 + beta_2 <= 2:
                sample = np.random.rand()
                if sample <= (1 - beta_2) / ((1 - beta_1) + (1 - beta_2)):
                    frac_copy[i] = 1
                else:
                    frac_copy[j] = 1

            if frac_copy[i] == 0:
                frac_copy[j] = beta_1 + beta_2
                fracs_indicies = np.delete(fracs_indicies, 0)

            if frac_copy[i] == 1:
                frac_copy[j] = beta_2 - (1 - beta_1)
                fracs_indicies = np.delete(fracs_indicies, 0)

            if frac_copy[j] == 0:
                frac_copy[i] = beta_1 + beta_2
                fracs_indicies = np.delete(fracs_indicies, 1)
            if frac_copy[j] == 1:
                frac_copy[i] = beta_1 - (1 - beta_2)
                fracs_indicies = np.delete(fracs_indicies, 1)

        return frac_copy

    def dep_round_2(self, frac):
        frac_copy = copy.copy(frac)
        fracs_indicies = np.where(np.logical_and(frac_copy > 0, frac_copy < 1))[0]
        while np.size(fracs_indicies) >= 2:  # there exist at least one fractional element
            i = fracs_indicies[0]
            j = fracs_indicies[1]
            beta_1 = frac_copy[i]
            beta_2 = frac_copy[j]

            if 0 <= self.weights[i] * beta_1 + self.weights[j] * beta_2 <= min(self.weights[i], self.weights[j]):
                sample = np.random.rand()
                if sample <= (self.weights[j] * beta_2) / (self.weights[i] * beta_1 + self.weights[j] * beta_2):
                    frac_copy[i] = 0
                else:
                    frac_copy[j] = 0

            elif self.weights[i] <= self.weights[i] * beta_1 + self.weights[j] * beta_2 <= self.weights[j]:
                sample = np.random.rand()
                if sample <= beta_1:
                    frac_copy[i] = 1
                else:
                    frac_copy[i] = 0

            elif self.weights[j] <= self.weights[i] * beta_1 + self.weights[j] * beta_2 <= self.weights[i]:
                sample = np.random.rand()
                if sample <= beta_2:
                    frac_copy[j] = 1
                else:
                    frac_copy[j] = 0

            elif max(self.weights[i], self.weights[j]) <= self.weights[i] * beta_1 + self.weights[j] * beta_2 <= self.weights[i] + self.weights[j]:
                sample = np.random.rand()
                if sample <= (self.weights[j] * (1 - beta_2)) / (self.weights[i] * (1 - beta_1) + self.weights[j] * (1 - beta_2)):
                    frac_copy[i] = 1
                else:
                    frac_copy[j] = 1
        #####################
            if frac_copy[i] == 0:
                frac_copy[j] = beta_1 * (self.weights[i]/self.weights[j]) + beta_2
                fracs_indicies = np.delete(fracs_indicies, 0)

            if frac_copy[i] == 1:
                frac_copy[j] = beta_2 - (1 - beta_1) * (self.weights[i]/self.weights[j])
                fracs_indicies = np.delete(fracs_indicies, 0)

            if frac_copy[j] == 0:
                frac_copy[i] = beta_1 + beta_2 * (self.weights[j]/self.weights[i])
                fracs_indicies = np.delete(fracs_indicies, 1)
            if frac_copy[j] == 1:
                frac_copy[i] = beta_1 - (1 - beta_2) * (self.weights[j]/self.weights[i])
                fracs_indicies = np.delete(fracs_indicies, 1)

            # fracs_indicies = np.where(np.logical_and(frac_copy > 0, frac_copy < 1))[0]

        return frac_copy



