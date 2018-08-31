import numpy as np
from scipy.special import gammaln

def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

def mse(y_true, y_pred):
    return (y_true - y_pred) ** 2

def poisson(y_true, y_pred):
    return y_pred - y_true * np.log(y_pred + 1e-07) + gammaln(y_true + 1)
