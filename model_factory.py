from keras.models import Sequential
from keras import layers

class NaiveModel():
    pass

class SimpleRNNmodel:
    def build_model(self, float_data, optimizer, loss):
        model = Sequential()
        model.add(layers.SimpleRNN(32,
                                   input_shape=(None, float_data.shape[-1])))
        model.add(layers.Dense(258, activation='relu'))
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
        model.add(layers.Dense(258))
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
        model.add(layers.Dense(258))
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
        model.add(layers.Dense(258))
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
        model.add(layers.Dense(258))
        model.compile(optimizer=optimizer, loss=loss)
        return model
