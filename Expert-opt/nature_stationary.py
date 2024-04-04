import numpy as np

class CN():
    def __init__(self, files=1000):
        self.files = files

        self.grad = 0
        self.ind = -1
        self.min = np.uint64(1)
        self.max = np.uint64(self.files)

        self.stat_trace = np.random.zipf(1.01, 10000) # This is T
        self.stat_trace = (self.stat_trace / float(max(self.stat_trace))) * (self.files - 1) # times the id of the largest file.

    def generate_cost(self):  # this is generating r
        # requested_id = self.Zipf(1.05, self.min, self.max) - 1
        self.ind = self.ind + 1
        requested_id = self.stat_trace[self.ind]
        requested_id = int(requested_id)
        r = np.zeros(self.files)
        r[requested_id] = 1
        self.grad = r
        return  self.grad