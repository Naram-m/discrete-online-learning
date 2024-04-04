import numpy as np

class NatureMLDS():
    def __init__(self, files=28478, max_cache_size=500):
        # self.trace = np.load("./ml_trace_10k.npy")
        self.trace = np.load("./ml_trace_10k_mini.npy")
        # self.files = 28478
        self.files = 10379
        self.max_cache_size = max_cache_size
        self.grad = 0
        self.ind = -1

    def generate_cost(self):  # this is generating r
        self.ind += 1
        requested_id = self.trace[self.ind]
        # print("Requesting ", requested_id)
        requested_id = int(requested_id)
        r = np.zeros(self.files)
        r[requested_id] = 1
        self.grad = r
        # if you have utility, multiply here
        return self.grad
