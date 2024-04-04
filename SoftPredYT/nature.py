import numpy as np
from yt_data_interface import DataInterface

class NatureYTDS():
    def __init__(self, files=28478, max_cache_size=500):
        self.di = DataInterface()
        self.files = 10000
        self.max_cache_size = max_cache_size
        self.grad = 0

    def generate_cost(self):  # this is generating r
        requested_id = self.di.tr_sample()
        # print("Requesting ", requested_id)
        requested_id = int(requested_id)
        r = np.zeros(self.files)
        r[requested_id] = 1
        self.grad = r
        # if you have utility, multiply here
        return self.grad
