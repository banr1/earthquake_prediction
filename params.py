import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--lookback', type=int, default=365)
arg('--step', type=int, default=1)
arg('--delay', type=int, default=93)
arg('--batch_size', type=int, default=30)
arg('--optimizer', default='SGD')
    #'SGD'
    #'RMSprop'
    #'Adam'
    #'Adagrad'
    #'Adadelta'
    #'Nadam'
arg('--loss', default='mae')
    #'mae'
    #'mse'
arg('--epochs', type=int, default=20)
arg('--start_day', default='2000-02-01')
arg('--end_day', default='2016-01-31')
arg('--train_max_idx', type=int, default=4000)
arg('--val_max_idx', type=int, default=5000)
arg('--test_max_idx', type=int, default=None)
arg('--model', default='SimpleRNNmodel')
    #'SimpleRNNmodel'
    #'GRUmodel'
    #'StackingGRUmodel'
    #'LSTMmodel'
    #'StackingLSTMmodel'

args = parser.parse_args()
