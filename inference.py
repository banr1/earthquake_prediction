import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime
from PIL import Image
import keras.optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from params import args
import model_factory

if __name__ == '__main__':
    date_format = '%Y-%m-%d'
    np.random.seed(42)

def strptime(str):
    return datetime.datetime.strptime(str, date_format)

def get_period(start_day, split_day_1, split_day_2, end_day):
    train_start = start_day
    train_end = split_day_1 - datetime.timedelta(days=1)
    train_period = (train_end - train_start).days + 1
    val_start = split_day_1
    val_end = split_day_2 - datetime.timedelta(days=1)
    val_period = (val_end - val_start).days + 1
    test_start = split_day_2
    test_end = end_day
    test_period = (test_end - test_start).days + 1
    print('train    period: {} ~ {} ({:0=4}days)'.format(train_start.strftime(date_format),
                                                         train_end.strftime(date_format),
                                                         train_period))
    print('validate period: {} ~ {} ({:0=4}days)'.format(val_start.strftime(date_format),
                                                         val_end.strftime(date_format),
                                                         val_period))
    print('test     period: {} ~ {} ({:0=4}days)'.format(test_start.strftime(date_format),
                                                         test_end.strftime(date_format),
                                                         test_period))
    return train_period, val_period, test_period

def find_class_by_name(name, modules):
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def raw_to_csv(raw_files, csv_file):
    with open(csv_file, 'w') as csv:
        header = "year,month,day,longitude,latitude,depth,magnitude\n"
        csv.writelines(header)
        for raw_file in raw_files:
            with open(raw_file, 'r') as raw:
                raw_lines = raw.readlines()
                for raw_line in raw_lines:
                    if raw_line[0] != 'J':
                        continue
                    elif raw_line[52] in [' ', '-', 'A', 'B', 'C']:
                        continue
                    year = raw_line[1:5]
                    month = raw_line[5:7]
                    day = raw_line[7:9]
                    latitude = raw_line[22:24]
                    longitude = raw_line[33:36]
                    depth = raw_line[45:49].strip()
                    magnitude = raw_line[52:54]
                    csv_list = [year, month, day, longitude, latitude, depth, magnitude]
                    csv_line = ", ".join(csv_list) + "\n"
                    csv.writelines(csv_line)

def get_grid_data(df):
    df = df[df['latitude'] >= 30] #302
    df = df[df['latitude'] <= 46] #456
    df = df[df['longitude'] >= 129] #1285
    df = df[df['longitude'] <= 147] #1470
    df['latlon'] = df['latitude'].astype(str) + '-' + df['longitude'].astype(str)
    df = df[['year', 'month', 'day', 'latlon']]
    df = df.set_index('latlon')
    df = df.reset_index()
    return df

def get_daily_data(df, start, end, dummy_col):
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'latlon']]
    df = pd.get_dummies(df, columns=['latlon'], prefix='', prefix_sep='')
    col = pd.DataFrame(columns=dummy_col)
    df = pd.concat([col, df], join='outer') #以前は、pd.merge(col, df, how='outer') --> df.sort_index(axis=1)
    df = df.fillna(0)
    df = df.groupby('time').sum()
    df = df[start: end] #終了日も含む
    idx = pd.DataFrame(index=pd.date_range(start, end))
    df = pd.merge(idx, df, how='outer', left_index=True, right_index=True)
    df = df.fillna(0)
    df = df.astype(int)
    return df

def minmax_scaling(float_data, end_idx):
    max = float_data[:end_idx].max(axis=(0,1))
    print("max:{}".format(max))
    float_data /= max
    return float_data

def pseudo_standarization(float_data, end_idx):
    mean = 0
    std = float_data[:end_idx].std(axis=(0,1))
    print('mean:{}, std:{}'.format(mean, std))
    float_data = (float_data - mean) / std
    return float_data, mean, std

def destandarization(float_data, mean, std):
    return float_data * std + mean

def generator(data, lookback, min_idx, max_idx, batch_size, target_length, shuffle=False):
    if max_idx is None:
        max_idx = len(data) - 1
    i = min_idx + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_idx + lookback, max_idx, size=batch_size)
        else:
            if i + batch_size >= max_idx:
                i = min_idx + lookback
            rows = np.arange(i, min(i + batch_size, max_idx))
            i += len(rows)
        samples = np.zeros((len(rows), lookback, data.shape[-1]))
        targets = np.zeros((len(rows), target_length))
        for j, row in enumerate(rows):
            idxs = range(rows[j] - lookback, rows[j])
            samples[j] = data[idxs]
            targets[j] = data[rows[j]][-target_length:]
        yield samples, targets

def train_and_validate(float_data, dummy_col, target_length, train_period, val_period, test_period, mean, std):
    lookback = args.lookback
    batch_size = args.batch_size
    optimizer_class = find_class_by_name(args.optimizer, [keras.optimizers])
    train_gen = generator(float_data,
                          lookback=lookback,
                          min_idx=0,
                          max_idx=train_period,
                          batch_size=batch_size,
                          target_length=target_length,
                          shuffle=args.train_shuffle)
    val_gen = generator(float_data,
                        lookback=lookback,
                        min_idx=train_period + 1,
                        max_idx=train_period + val_period,
                        batch_size=batch_size,
                        target_length=target_length)
    test_gen = generator(float_data,
                         lookback=lookback,
                         min_idx=train_period + val_period + 1,
                         max_idx=None,
                         batch_size=batch_size,
                         target_length=target_length)
    train_steps = (train_period - lookback) // batch_size
    val_steps = (val_period - 1 - lookback) // batch_size
    test_steps = (len(float_data) - (train_period + val_period + 1) - lookback) // batch_size
    model = find_class_by_name(args.model, [model_factory])().build_model(float_data,
                                                                          lookback=lookback,
                                                                          batch_size=batch_size,
                                                                          optimizer=optimizer_class(),
                                                                          loss=args.loss,
                                                                          stateful=args.stateful, 
                                                                          target_length=target_length)
    model.summary()
    callbacks = [ModelCheckpoint(filepath=args.log_dir + 'my_model.h5'),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
                 TensorBoard(log_dir=args.log_dir + 'tensorboard/')]
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps // args.train_step_ratio,
                                  epochs=args.epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  verbose=1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b-', label='Training loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation loss')
    plt.ylim(ymin=0)
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(args.log_dir + 'loss_{}.png'.format(args.model))
    pred = model.predict_generator(test_gen,
                                   steps=test_steps,
                                   verbose=1)
    pred = pred[-92:, :]
    naive_pred = float_data[-549: -457, -259:]
    true = float_data[-92:, -259:]
    pred = destandarization(pred, mean, std)
    naive_pred = destandarization(naive_pred, mean, std)
    true = destandarization(true, mean, std)
    df_pred = pd.DataFrame(pred).sum(axis=0)
    df_naive_pred = pd.DataFrame(naive_pred).sum(axis=0)
    df_true = pd.DataFrame(true).sum(axis=0)
    df_eval = pd.concat([df_naive_pred, df_pred, df_true], axis=1)
    df_eval.columns = ['na_pred', 'pred', 'true']
    df_eval['na_eval'] = np.abs(df_eval['na_pred'] - df_eval['true'])
    df_eval['eval'] = np.abs(df_eval['pred'] - df_eval['true'])
    df_eval = df_eval.loc[:, ['true', 'pred', 'eval', 'na_pred', 'na_eval']]
    arr_print = df_eval.sort_values(by='eval').values
    arr_print = np.concatenate([arr_print, arr_print.sum(axis=0).reshape(1, -1)])
    print(pd.DataFrame(arr_print[-20:]))
    naive_eval = np.mean(df_eval['na_eval'])
    eval = np.mean(df_eval['eval'])
    print('[eval({})] {}:{:.5f}, Naivemodel:{:.5f}'.format(args.loss, args.model, eval, naive_eval))
    df_pred.to_csv(args.log_dir + 'eval_{}.csv'.format(args.model, eval))
    #evaluate = model.evaluate_generator(test_gen,
    #                                    steps=test_steps,
    #                                    verbose=1)
    #print(evaluate)

def main():
    start_day = strptime(args.start_day)
    split_day_1 = strptime(args.split_day_1)
    split_day_2 = strptime(args.split_day_2)
    end_day = strptime(args.end_day)
    train_period, val_period, test_period = get_period(start_day, split_day_1, split_day_2, end_day)
    raw_files = sorted(glob.glob(args.input_raw_dir + 'h????'))
    csv_file = args.input_preprocessed_dir + 'df.csv'
    if not os.path.exists(csv_file):
        raw_to_csv(raw_files, csv_file)
    df = pd.read_csv(csv_file, low_memory=False)
    df_m2 = df[df['magnitude'] >= 20]
    df_m2 = get_grid_data(df_m2)
    latlon = np.sort(df_m2['latlon'].unique())
    df_m2 = get_daily_data(df_m2, start=args.start_day, end=args.end_day, dummy_col=latlon)
    df_m4 = df[df['magnitude'] >= 40]
    df_m4 = get_grid_data(df_m4)
    df_m4 = get_daily_data(df_m4, start=args.start_day, end=args.end_day, dummy_col=latlon)
    float_data_m2 = df_m2.values.astype(np.float64)
    float_data_m4 = df_m4.values.astype(np.float64)
    float_data_m2, _, _ = pseudo_standarization(float_data_m2, train_period)
    float_data_m4, mean, std = pseudo_standarization(float_data_m4, train_period)
    plt.hist(float_data_m2[:train_period].sum(axis=0), bins=10)
    plt.savefig('ignore_hist.png')
    target_length = float_data_m4.shape[1]
    float_data = np.hstack([float_data_m2, float_data_m4])
    print('float_data shape: {}'.format(float_data.shape))
    train_and_validate(float_data, latlon, target_length, train_period, val_period, test_period, mean, std)
    return

if __name__ == '__main__':
    main()
