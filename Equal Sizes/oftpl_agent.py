import numpy as np


class O_FTPLAgent ():
    def __init__(self, files, max_cache_size, acc):
        self.files = files
        self.max_cache_size = max_cache_size
        self.acc_grads = np.zeros(self.files)
        self.perturbation = np.random.normal(0, 1, files)
        self.eta = 0
        self.acc_norm = 0
        self.acc=acc
        # self.const = 1 / np.sqrt(self.max_cache_size) * (1 / (np.log(self.files) * np.pi))**(1/4)
        # self.const = 1.3 / np.sqrt(self.max_cache_size) * (1/np.log(self.files * np.exp(1) / self.max_cache_size))**(1/4)
        self.const = 1 / np.sqrt(self.max_cache_size) * (1/np.log(self.files * np.exp(1) / self.max_cache_size))**(1/4)

    def step(self, grad):
        ## fresh gamma
        self.perturbation = np.random.normal(0, 1, self.files)

        if np.random.rand() <= self.acc:
            pred = grad
        else:
            pred = np.zeros(self.files)
            pred [np.random.randint(0, self.files)] = 1
        self.eta = self.const * np.sqrt(self.acc_norm)
        to_be_used = self.acc_grads + pred + self.eta * self.perturbation
        ind = np.argpartition(to_be_used, -self.max_cache_size)[
              -self.max_cache_size:]  # indicies of the most # requested
        y = np.zeros(self.files)
        y[ind] = 1
        self.acc_grads += grad
        self.acc_norm += np.linalg.norm(grad - pred, 1)**2
        return y