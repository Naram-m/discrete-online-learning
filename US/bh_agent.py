import numpy as np
from knap_solver import ks

class BHSAgent ():
    def __init__(self, files, max_cache_size):
        self.files = files
        self.max_cache_size = max_cache_size
        self.acc_grads = np.zeros(self.files)
        self.weights = np.load("./weights.npy")

    def step(self, grad):
        self.acc_grads += grad
        y = ks(self.acc_grads, self.weights, self.max_cache_size)[1]
        return y
