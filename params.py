import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--input_raw_dir', default='../input/raw/')
arg('--input_preprocessed_dir', default='../input/preprocessed/')
arg('--log_dir', default='../log/')
arg('--lookback', type=int, default=365)
arg('--step', type=int, default=1)
arg('--delay', type=int, default=92) #11月: 30days, 12月: 31days, 1月: 31days
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
arg('--start_day', default='2000-01-01')
arg('--split_day', default='2014-11-01')
arg('--end_day', default='2016-01-31')
arg('--model', default='SimpleRNNmodel')
    #'SimpleRNNmodel'
    #'GRUmodel'
    #'StackingGRUmodel'
    #'LSTMmodel'
    #'StackingLSTMmodel'

args = parser.parse_args()
