import copy
import numpy as np
# import cplex
import cvxpy as cp

class RobustExpert():
    def __init__(self, files, max_cache_size):
        self.files = files
        self.y_dim = self.files
        # self.z = cp.Variable(self.z_dim)
        self.y = cp.Variable(self.y_dim)
        # self.x = cp.hstack([self.z, self.y])
        # self.acc_grads = np.zeros(self.z_dim)
        self.acc_grads = np.zeros(self.y_dim)

        self.R = np.sqrt(max_cache_size)  # an upper bound on the l2 norm of x

        self.max_cache_size = max_cache_size
        self.standard_constraints = []
        self.actions = []
        self.grads = []

        ####################################### Constraints ###################################
        # For y:
        self.standard_constraints.extend([self.y >= 0, self.y <= 1])
        self.standard_constraints.extend([cp.sum(self.y) == self.max_cache_size])
        self.non_projected_sol_parameter = cp.Parameter(self.y_dim)
        self.objective = cp.Minimize(cp.sum_squares(self.y - self.non_projected_sol_parameter))
        self.prob = cp.Problem(self.objective, self.standard_constraints)


    def step(self, grad):
        if not self.actions:
            yy = np.random.rand(self.y_dim)
        else:
            ogd_step = np.sqrt(2) * self.R / (1 * np.sqrt(len(self.grads)))
            yy = self.actions[-1] + ogd_step * self.grads[-1]          # we are returning the reward

        self.non_projected_sol_parameter.value = yy
        result = self.prob.solve(warm_start=True)
        self.grads.append(grad)
        self.actions.append(self.y.value)
        return self.y.value





