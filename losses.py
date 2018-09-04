import numpy as np
import keras.backend as K
from tensorflow import lgamma
from scipy.special import gammaln

def absolute_error(y_true, y_pred):
    return np.abs(y_true - y_pred)

def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2

def poisson_log_likelihood(y_true, y_pred):
    return y_pred - y_true * np.log(y_pred + 1e-07) + gammaln(y_true + 1)

def mean_poisson_log_likelihood(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()) + lgamma(y_true + 1), axis=-1)
