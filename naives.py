import numpy as np

class Naive():
    def __init__(self, mean, batch_size, target_length):
        self.lam = mean
        self.batch_size = batch_size
        self.target_length = target_length

    def predict(self, samples):
        e = 1e-5
        pred = np.random.poisson(lam=self.lam-e, size=(self.batch_size, self.target_length))
        return pred + e

class Basenaive():
    def __init__(self):
        pass

class Poissonnaive(Basenaive):
    def build_naive(self, float_data, batch_size, target_length):
        mean = np.mean(float_data[-target_length:], axis=(0,1))
        naive = Naive(mean, batch_size, target_length)
        return naive
