import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--model', default='SimpleRNNmodel')
    #'SimpleRNNmodel', 'GRUmodel', 'StackingGRUmodel', 'LSTMmodel', 'StackingLSTMmodel'
arg('--optimizer', default='SGD')
    #'SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta', 'Nadam'
arg('--loss', default='general_poisson')
    #'mae', 'mse', 'poisson', 'general_poisson'
arg('--stateful', type=bool, default=True)
arg('--lookback', type=int, default=365)
arg('--batch_size', type=int, default=31)
arg('--epochs', type=int, default=15)
arg('--train_shuffle', type=bool, default=False)
arg('--train_step_ratio', type=int, default=1)
arg('--start_day', default='1998-02-01')
arg('--split_day_1', default='2010-02-01')
arg('--split_day_2', default='2014-10-01')
arg('--end_day', default='2016-01-31')
arg('--input_raw_dir', default='../input/raw/')
arg('--input_preprocessed_dir', default='../input/preprocessed/')
arg('--log_dir', default='../log/')

args = parser.parse_args()
