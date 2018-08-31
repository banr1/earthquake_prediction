import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--input_raw_dir', default='../input/raw/')
arg('--input_preprocessed_dir', default='../input/preprocessed/')
arg('--log_dir', default='../log/')
arg('--lookback', type=int, default=365)
arg('--batch_size', type=int, default=30)
arg('--train_shuffle', type=bool, default=False)
arg('--optimizer', default='SGD')
    #'SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta', 'Nadam'
arg('--loss', default='poisson')
    #'mae', 'mse', 'poisson'
arg('--stateful', type=bool, default=True)
arg('--train_step_ratio', type=int, default=1)
arg('--epochs', type=int, default=15)
arg('--start_day', default='1998-02-01')
arg('--split_day_1', default='2009-02-01')
arg('--split_day_2', default='2012-02-01')
arg('--end_day', default='2016-01-31')
arg('--model', default='SimpleRNNmodel')
    #'SimpleRNNmodel', 'GRUmodel', 'StackingGRUmodel', 'LSTMmodel', 'StackingLSTMmodel'

args = parser.parse_args()
