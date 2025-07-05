import numpy as np


class RandomDistributionReform:
    def __init__(self, array, args: tuple):
        self.array = array
        self.args = args
        if len(self.array.shape) > 1:
            self.shape = (self.array.shape[1], self.array.shape[1])
        else:
            self.shape = self.array.shape

    def beta_distribution(self):
        beta = np.random.beta(a=self.args[0], b=self.args[1], size=self.shape)
        return np.dot(self.array, beta)

    def gamma_distribution(self):
        gamma = np.random.gamma(shape=self.args[0], scale=self.args[1], size=self.shape)
        return np.dot(self.array,gamma)
