import keras.backend as K
from tensorflow import lgamma

def general_poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()) + lgamma(y_true + 1), axis=-1)
