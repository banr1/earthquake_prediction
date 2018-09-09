import argparse
import datetime

def boolian(str):
    if str in ('True'):
        return True
    elif str in ('False'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))

def intlist(str):
    return [int(i) for i in str.split(',')]

def floatlist(str):
    return [float(i) for i in str.split(',')]

def day(str):
    return datetime.datetime.strptime(str, '%Y-%m-%d')

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--model', default='SimpleRNNmodel')
    #ex) 'SimpleRNNmodel', 'GRUmodel', 'StackedGRUmodel', 'LSTMmodel', 'StackedLSTMmodel'
arg('--version', type=int, default='99')
arg('--optimizer', default='SGD')
    #ex) 'SGD', 'RMSprop', 'Adam', 'Adagrad', 'Adadelta', 'Nadam'
arg('--loss', default='mean_poisson_log_likelihood')
    #ex) 'mean_squared_error', 'mean_absolute_error', 'mean_poisson_log_likelihood'
arg('--stateful', type=boolian, default='True')
arg('--num_layers', type=intlist, default='32')
arg('--dropouts', type=floatlist, default='0.')
arg('--recurrent_dropouts', type=floatlist, default='0.')
arg('--lookback', type=int, default='365')
arg('--batch_size', type=int, default='31')
arg('--epochs', type=int, default='15')
arg('--train_shuffle', type=boolian, default='False')
arg('--random_seed', type=int, default='42')
arg('--naive_period', type=int, default='1')
arg('--start_day', type=day, default='1998-02-01')
arg('--split_day_1', type=day, default='2010-02-01')
arg('--split_day_2', type=day, default='2014-10-01')
arg('--end_day', type=day, default='2016-01-31')
arg('--input_raw_dir', default='../input/raw/')
arg('--input_preprocessed_dir', default='../input/preprocessed/')
arg('--log_dir', default='../log/')
arg('--record', type=boolian, default='False')

args = parser.parse_args()
