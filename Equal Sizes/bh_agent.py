import numpy as np


class BHSAgent ():
    def __init__(self, files, max_cache_size):
        self.files = files
        self.max_cache_size = max_cache_size
        self.acc_grads = np.zeros(self.files)

    def step(self, grad):
        self.acc_grads += grad
        ind = np.argpartition(self.acc_grads, -self.max_cache_size)[
              -self.max_cache_size:]  # indicies of the most # requested
        y = np.zeros(self.files)
        y[ind] = 1
        return y
