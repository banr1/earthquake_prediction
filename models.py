import numpy as np
from keras.models import Model
from keras import Input
from keras.layers import Dense, SimpleRNN, GRU, LSTM

class Basemodel():
    def __init__(self):
        pass

class SimpleRNNmodel(Basemodel):
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length,
                    num_filters, dropouts, recurrent_dropouts):
        if stateful:
            input = Input(batch_shape=(batch_size, lookback, float_data.shape[-1]))
        else:
            input = Input(shape=(None, float_data.shape[-1]))
        x = SimpleRNN(num_filters[0], activation='relu', stateful=stateful,
                      dropout=dropouts[0], recurrent_dropout=recurrent_dropouts[0])(input)
        output = Dense(target_length, activation='relu')(x)
        model = Model(input, output)
        model.compile(optimizer=optimizer, loss=loss)
        return model

class GRUmodel(Basemodel):
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length,
                    num_filters, dropouts, recurrent_dropouts):
        if stateful:
            input = Input(batch_shape=(batch_size, lookback, float_data.shape[-1]))
        else:
            input = Input(shape=(None, float_data.shape[-1]))
        x = GRU(num_filters[0], activation='tanh', stateful=stateful,
                dropout=dropouts[0], recurrent_dropout=recurrent_dropouts[0])(input)
        output = Dense(target_length, activation='relu')(x)
        model = Model(input, output)
        model.compile(optimizer=optimizer, loss=loss)
        return model

class LSTMmodel(Basemodel):
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length,
                    num_filters, dropouts, recurrent_dropouts):
        if stateful:
            input = Input(batch_shape=(batch_size, lookback, float_data.shape[-1]))
        else:
            input = Input(shape=(None, float_data.shape[-1]))
        x = LSTM(num_filters[0], activation='tanh', stateful=stateful,
                 dropout=dropouts[0], recurrent_dropout=recurrent_dropouts[0])(input)
        output = Dense(target_length, activation='relu')(x)
        model = Model(input, output)
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackedGRUmodel(Basemodel):
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length,
                    num_filters, dropouts, recurrent_dropouts):
        if stateful:
            input = Input(batch_shape=(batch_size, lookback, float_data.shape[-1]))
        else:
            input = Input(shape=(None, float_data.shape[-1]))
        x = GRU(num_filters[0], activation='tanh', stateful=stateful, return_state=True,
                dropout=dropouts[0], recurrent_dropout=recurrent_dropouts[0])(input)
        x = GRU(num_filters[1], activation='relu', stateful=stateful, return_state=False,
                dropout=dropouts[1], recurrent_dropout=recurrent_dropouts[1])(input)
        output = Dense(target_length, activation='relu')(x)
        model = Model(input, output)
        model.compile(optimizer=optimizer, loss=loss)
        return model

class StackedLSTMmodel(Basemodel):
    def build_model(self, float_data, lookback, batch_size, optimizer, loss, stateful, target_length,
                    num_filters, dropouts, recurrent_dropouts):
        if stateful:
            input = Input(batch_shape=(batch_size, lookback, float_data.shape[-1]))
        else:
            input = Input(shape=(None, float_data.shape[-1]))
        x = LSTM(num_filters[0], activation='tanh', stateful=stateful, return_state=True,
                dropout=dropouts[0], recurrent_dropout=recurrent_dropouts[0])(input)
        x = LSTM(num_filters[1], activation='relu', stateful=stateful, return_state=False,
                dropout=dropouts[1], recurrent_dropout=recurrent_dropouts[1])(input)
        output = Dense(target_length, activation='relu')(x)
        model = Model(input, output)
        model.compile(optimizer=optimizer, loss=loss)
        return model
