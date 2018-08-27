import numpy as np
from keras.models import Sequential
from keras import layers

class NaiveModel():
    def build_model(self, test_gen, test_steps, target_length):
        batch_losses = []
        for step in range(test_steps):
            samples, targets = next(test_gen)
            preds = samples[:, -1, target_length:]
            loss = np.mean(np.abs(preds - targets))
            batch_losses.append(loss)
        print(np.mean(batch_losses))
        return

class SimpleRNNmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.SimpleRNN(32,
                                   activation='relu',
                                   input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(259, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class GRUmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.GRU(32,
                             dropout=0.2,
                             recurrent_dropout=0.2,
                             return_sequences=False,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(259, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackingGRUmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.GRU(32,
                             dropout=0.1,
                             recurrent_dropout=0.5,
                             return_sequences=True,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.GRU(64,
                             activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(259, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class LSTMmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.LSTM(32,
                             dropout=0.2,
                             recurrent_dropout=0.2,
                             return_sequences=False,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(259, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackingLSTMmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.LSTM(32,
                             dropout=0.1,
                             recurrent_dropout=0.5,
                             return_sequences=True,
                             input_shape=(None, float_data.shape[-1])))
        model.add(layers.LSTM(64,
                             activation='relu',
                             dropout=0.1,
                             recurrent_dropout=0.5))
        model.add(layers.Dense(259, activation='relu'))
        model.compile(optimizer=optimizer, loss=loss)
        return model
