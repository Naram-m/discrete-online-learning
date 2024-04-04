from scipy import io
import numpy as np

class DataInterface():
    def __init__(self):
        np.random.seed(42)
        loaded = io.loadmat('./yt_trace.mat')
        print(sorted(loaded.keys()))
        np_trace = loaded['trace0']

        (unique, counts) = np.unique(np_trace[0], return_counts=True)
        frequencies = np.asarray((unique, counts)).T

        self.clipped = frequencies[frequencies[:,1] >= 2]
        # self.clipped = frequencies[frequencies[:,1] >= 3]
        self.clipped = self.clipped[self.clipped[:, 1].argsort()[::-1]]
        self.clipped[:, 0] = np.arange(self.clipped.shape[0])
        self.clipped = self.clipped[0:10000,:]
        self.denom = sum(self.clipped[:, 1])


    def tr_sample(self):
        s = np.random.choice(self.clipped[:, 0], p=self.clipped[:, 1]/self.denom)
        return s

