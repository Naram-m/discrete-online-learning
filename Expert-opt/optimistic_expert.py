import numpy as np
import copy
import cvxpy as cp
import random

class OptimisticExpert:
    def __init__(self, files, max_cache_size, rho):
        self.files = files
        self.y_dim = self.files
        self.y = cp.Variable(self.y_dim)
        self.max_cache_size = max_cache_size
        self.rho = rho
        self.actions = []
        self.standard_constraints = []
        ####################################### Constraints ###################################
        # For y:
        # self.standard_constraints.extend([self.y >= 0, self.y <= 1])
        # self.standard_constraints.extend(
        #     [cp.sum(self.y) <= self.max_cache_size])
        #
        # self.grad_parameter = cp.Parameter(self.y_dim, nonneg=True)
        # self.objective = cp.Maximize(self.grad_parameter @ self.y)
        # self.prob = cp.Problem(self.objective, self.standard_constraints)
        self.ind = -1
        self.new_y = np.zeros(self.files)

    def step(self, grad):
        self.ind = self.ind + 1
        if self.ind == 0:
            # ind = np.random.randint(0, self.files, self.max_cache_size)
            ind = random.sample(range(self.files), self.max_cache_size)
            self.new_y[ind] = 1
        else:
            if np.random.rand() <= self.rho:
                pred_index = np.where(grad == 1)[0][0]
            else:
                pred_index = np.random.randint(0, self.files, 1)

            if self.new_y[pred_index] != 1: # replace only if needed
                ones = np.where(self.new_y == 1)[0]
                self.new_y[np.random.choice(ones)] = 0
                self.new_y[pred_index] = 1

        # print("Sum: ", np.sum(self.new_y))
        return self.new_y

        # self.grad_parameter.value = pred
        # result = self.prob.solve(warm_start=True, solver='ECOS')
        # self.y.value[self.y.value < 0.1] = 0      #!!!!!!!!!!!!!!!!!!!!!!!
        # return self.y.value
