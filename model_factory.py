import numpy as np
from keras.models import Sequential
from keras import layers

class SimpleRNNmodel:
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length):
        model = Sequential()
        if stateful:
            model.add(layers.SimpleRNN(32,
                                       activation='relu',
                                       stateful=stateful,
                                       batch_input_shape=(batch_size, lookback, float_data.shape[-1])))
        else:
            model.add(layers.SimpleRNN(32,
                                       activation='relu',
                                       stateful=stateful,
                                       input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(target_length, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class GRUmodel:
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length):
        model = Sequential()
        if stateful:
            model.add(layers.GRU(32,
                                 activation='tanh',
                                 stateful=stateful,
                                 batch_input_shape=(batch_size, lookback, float_data.shape[-1])))
        else:
            model.add(layers.GRU(32,
                                 activation='tanh',
                                 stateful=stateful,
                                 input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(target_length, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackingGRUmodel:
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length):
        model = Sequential()
        if stateful:
            model.add(layers.GRU(32,
                                 activation='tanh',
                                 dropout=0.1,
                                 recurrent_dropout=0.5,
                                 return_sequences=True,
                                 stateful=stateful,
                                 batch_input_shape=(batch_size, lookback, float_data.shape[-1])))
        else:
            model.add(layers.GRU(32,
                                 activation='tanh',
                                 dropout=0.1,
                                 recurrent_dropout=0.5,
                                 return_sequences=True,
                                 stateful=stateful,
                                 input_shape=(None, float_data.shape[-1])))
        model.add(layers.GRU(64,
                             activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(target_length, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class LSTMmodel:
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length):
        model = Sequential()
        if stateful:
            model.add(layers.LSTM(32,
                                  activation='tanh',
                                  dropout=0.2,
                                  recurrent_dropout=0.2,
                                  stateful=stateful,
                                  batch_input_shape=(batch_size, lookback, float_data.shape[-1])))
        else:
            model.add(layers.LSTM(32,
                                  activation='tanh',
                                  dropout=0.2,
                                  recurrent_dropout=0.2,
                                  stateful=stateful,
                                  batch_input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(target_length, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackingLSTMmodel:
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length):
        model = Sequential()
        if stateful:
            model.add(layers.LSTM(32,
                                  activation='tanh',
                                  dropout=0.1,
                                  recurrent_dropout=0.5,
                                  return_sequences=True,
                                  stateful=stateful,
                                  batch_input_shape=(batch_size, lookback, float_data.shape[-1])))
        else:
            model.add(layers.LSTM(32,
                                  activation='tanh',
                                  dropout=0.1,
                                  recurrent_dropout=0.5,
                                  return_sequences=True,
                                  stateful=stateful,
                                  input_shape=(None, float_data.shape[-1])))
        model.add(layers.LSTM(64,
                             activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(target_length, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model
