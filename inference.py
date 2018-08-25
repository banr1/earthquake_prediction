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

def get_period(start_day, split_day, end_day):
    train_start = start_day
    train_end = split_day - datetime.timedelta(days=1)
    train_period = (train_end - train_start).days + 1
    val_start = split_day
    val_end = end_day -  datetime.timedelta(days=args.delay)
    val_period = (val_end - val_start).days + 1
    test_start = val_end + datetime.timedelta(days=1)
    test_end = end_day
    test_period = (test_end - test_start).days + 1
    print('train period: {} ~ {} ({}days)'.format(train_start.strftime(date_format),
                                                  train_end.strftime(date_format),
                                                  train_period))
    print('validate period: {} ~ {} ({}days)'.format(val_start.strftime(date_format),
                                                     val_end.strftime(date_format),
                                                     val_period))
    print('test period: {} ~ {} ({}days)'.format(test_start.strftime(date_format),
                                                 test_end.strftime(date_format),
                                                 test_period))
    return train_period, val_period

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
    df = df.merge(col, how='outer')
    df = df.sort_index(axis=1)
    df = df.fillna(0)
    df = df.groupby('time').sum()
    df = df[start: end] #終了日も含む
    idx = pd.DataFrame(index=pd.date_range(start, end))
    df = pd.merge(idx, df, how='outer', left_index=True, right_index=True)
    df = df.fillna(0)
    df = df.astype(int)
    return df

def scaling(float_data, end_idx):  #min-max scaling
    max = float_data[:end_idx].max(axis=(0,1))
    print("max:{}".format(max))
    float_data /= max
    return float_data

def generator(data, lookback, delay, min_idx, max_idx, shuffle, batch_size, step):
    if max_idx is None:
        max_idx = len(data) - delay - 1
    i = min_idx + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_idx + lookback, max_idx, size=batch_size)
        else:
            if i + batch_size >= max_idx:
                i = min_idx + lookback
            rows = np.arange(i, min(i + batch_size, max_idx))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows), 258))
        for j, row in enumerate(rows):
            idxs = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[idxs]
            targets[j] = data[rows[j] + delay][258:]
        yield samples, targets

def train_and_validate(float_data, dummy_col, train_period, val_period):
    lookback = args.lookback
    step = args.step
    delay = args.delay
    batch_size = args.batch_size
    optimizer_class = find_class_by_name(args.optimizer, [keras.optimizers])
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_idx=0,
                          max_idx=train_period,
                          shuffle=False,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_idx=train_period + 1,
                        max_idx=val_period,
                        shuffle=False,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_idx=val_period + 1,
                         max_idx=None,
                         shuffle=False,
                         step=step,
                         batch_size=batch_size)
    train_steps = (train_period - lookback) // batch_size
    val_steps = (val_period - train_period - lookback) // batch_size
    test_steps = (len(float_data) - val_period - lookback) // batch_size
    if args.model == 'NaiveModel':
        batch_losses = []
        for step in range(val_steps):
            samples, targets = next(val_gen)
            preds = samples[:, -1, 258:]
            loss = np.mean(np.abs(preds - targets))
            batch_losses.append(loss)
        print(np.mean(batch_losses))
        return
    model = find_class_by_name(args.model, [model_factory])().build_model(float_data,
                                                                          optimizer=optimizer_class(),
                                                                          loss=args.loss)
    model.summary()
    callbacks = [
        ModelCheckpoint(filepath='my_model.h5'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)#,
        #TensorBoard(log_dir=args.log_dir, histogram_freq=1, embeddings_freq=1
        ]
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=args.epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  verbose=1)
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.ylim(ymin=0)
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig('loss_{}.png'.format(args.model))
    predict = model.predict_generator(test_gen,
                                      steps=test_steps,
                                      verbose=1)
    predict = predict.sum(axis=0)
    evaluate = model.evaluate_generator(test_gen,
                                        steps=test_steps,
                                        verbose=1)
    df_pred = pd.DataFrame(predict).T
    dummy_col = np.sort(dummy_col)
    df_pred.columns = dummy_col
    print(df_pred)
    print(evaluate)

def main():
    start_day = datetime.datetime.strptime(args.start_day, '%Y-%m-%d')
    split_day = datetime.datetime.strptime(args.split_day, '%Y-%m-%d')
    end_day = datetime.datetime.strptime(args.end_day, '%Y-%m-%d')
    train_period, val_period = get_period(start_day, split_day, end_day)
    raw_files = sorted(glob.glob(args.input_raw_dir + 'h????'))
    csv_file = args.input_preprocessed_dir + 'df.csv'
    if not os.path.exists(csv_file):
        raw_to_csv(raw_files, csv_file)
    df = pd.read_csv(csv_file)
    df_m2 = df[df['magnitude'] >= 20]
    df_m2 = get_grid_data(df_m2)
    latlon = df_m2['latlon'].unique()
    df_m2 = get_daily_data(df_m2, start=args.start_day, end=args.end_day, dummy_col=latlon)
    df_m4 = df[df['magnitude'] >= 40]
    df_m4 = get_grid_data(df_m4)
    df_m4 = get_daily_data(df_m4, start=args.start_day, end=args.end_day, dummy_col=latlon)
    print(df_m2.head())
    print(df_m4.head())
    float_data_m2 = df_m2.values.astype(np.float64)
    float_data_m4 = df_m4.values.astype(np.float64)
    float_data_m2 = scaling(float_data_m2, train_period)
    float_data_m4 = scaling(float_data_m4, train_period)
    float_data = np.hstack([float_data_m2, float_data_m4])
    print(float_data.shape)
    train_and_validate(float_data, latlon, train_period, val_period)
    return

if __name__ == '__main__':
    main()
