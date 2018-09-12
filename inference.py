import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime
import random
from PIL import Image
import keras.optimizers
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from params import args
import models
import losses

if __name__ == '__main__':
    model_name = args.model
    model_version = args.version
    optimizer_name = args.optimizer
    loss_name = args.loss
    stateful = args.stateful
    lookback = args.lookback
    batch_size = args.batch_size
    epochs = args.epochs
    num_layers = args.num_layers
    dropouts = args.dropouts
    recurrent_dropouts = args.recurrent_dropouts
    random_seed = args.random_seed
    naive_period = args.naive_period
    start_day = args.start_day
    split_day_1 = args.split_day_1
    split_day_2 = args.split_day_2
    end_day = args.end_day
    input_raw_dir = args.input_raw_dir
    input_preprocessed_dir = args.input_preprocessed_dir
    log_dir = args.log_dir
    record = args.record
    date_format = '%Y-%m-%d'
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.set_random_seed(random_seed)

def list_to_str(list):
    return ' '.join(map(str, list))

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
    df = df[df['latitude'] >= 30]
    df = df[df['latitude'] <= 46]
    df = df[df['longitude'] >= 129]
    df = df[df['longitude'] <= 147]
    df['latlon'] = df['latitude'].astype(str) + '-' + df['longitude'].astype(str)
    df = df[['year', 'month', 'day', 'latlon']]
    df = df.set_index('latlon')
    df = df.reset_index()
    return df

def get_daily_data(df, start, end, dummy_col):
    start = str(start)
    end = str(end)
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'latlon']]
    df = pd.get_dummies(df, columns=['latlon'], prefix='', prefix_sep='')
    col = pd.DataFrame(columns=dummy_col)
    df = pd.concat([col, df], join='outer', sort=False)
    df = df.fillna(0)
    df = df.groupby('time').sum()
    df = df[start: end] #終了日も含む
    idx = pd.DataFrame(index=pd.date_range(start, end))
    df = pd.merge(idx, df, how='outer', left_index=True, right_index=True)
    df = df.fillna(0)
    df = df.astype(int)
    return df

def naive_evaluate(test_gen, test_steps, pre_mean_loss, target_length, naive_period):
    errors = []
    for step in range(test_steps):
        samples, targets = next(test_gen)
        preds = samples[:, -naive_period, -target_length:]
        error = np.mean(pre_mean_loss(targets, preds))
        errors.append(error)
    return np.mean(errors)

def generator(data, lookback, min_idx, max_idx, batch_size, target_length):
    if max_idx is None:
        max_idx = len(data) - 1
    i = min_idx + lookback
    while 1:
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

def main():
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    train_period, val_period, test_period = get_period(start_day, split_day_1, split_day_2, end_day)
    raw_files = sorted(glob.glob(input_raw_dir + 'h????'))
    csv_file = input_preprocessed_dir + 'df.csv'
    if not os.path.exists(csv_file):
        raw_to_csv(raw_files, csv_file)

    df = pd.read_csv(csv_file, low_memory=False)
    df_m2 = df[df['magnitude'] >= 20]
    df_m2 = get_grid_data(df_m2)
    latlon = np.sort(df_m2['latlon'].unique())
    df_m2 = get_daily_data(df_m2, start=start_day, end=end_day, dummy_col=latlon)
    df_m4 = df[df['magnitude'] >= 40]
    df_m4 = get_grid_data(df_m4)
    df_m4 = get_daily_data(df_m4, start=start_day, end=end_day, dummy_col=latlon)
    float_data_m2 = df_m2.values.astype(np.float64)
    float_data_m4 = df_m4.values.astype(np.float64)
    max_m2 = float_data_m2.max(axis=(0,1))
    max_m4 = float_data_m4.max(axis=(0,1))
    float_data_m2 = float_data_m2 * max_m4 / max_m2
    plt.hist(float_data_m2[:train_period].sum(axis=0), bins=10)
    plt.savefig(log_dir + 'train_hist.png')
    target_length = float_data_m4.shape[1]
    float_data = np.hstack([float_data_m2, float_data_m4])
    print('float_data shape: {}'.format(float_data.shape))

    optimizer = find_class_by_name(optimizer_name, [keras.optimizers])()
    loss = find_class_by_name(loss_name, [losses, keras.losses])
    pre_mean_loss = find_class_by_name(loss_name.replace('mean_', ''), [losses])

    train_gen = generator(float_data,
                          lookback=lookback,
                          min_idx=0,
                          max_idx=train_period - 1,
                          batch_size=batch_size,
                          target_length=target_length)
    val_gen = generator(float_data,
                        lookback=lookback,
                        min_idx=train_period,
                        max_idx=train_period + val_period - 1,
                        batch_size=batch_size,
                        target_length=target_length)
    test_gen = generator(float_data,
                         lookback=lookback,
                         min_idx=train_period + val_period,
                         max_idx=None,
                         batch_size=batch_size,
                         target_length=target_length)

    train_steps = (train_period - lookback) // batch_size
    val_steps = (val_period - lookback) // batch_size
    test_steps = (len(float_data) - (train_period + val_period) - lookback) // batch_size
    model_class = find_class_by_name(model_name, [models])()
    model = model_class.build_model(float_data,
                                    lookback=lookback,
                                    batch_size=batch_size,
                                    optimizer=optimizer,
                                    loss=loss,
                                    stateful=stateful,
                                    target_length=target_length,
                                    num_layers=num_layers,
                                    dropouts=dropouts,
                                    recurrent_dropouts=recurrent_dropouts)
    model.summary()
    print('optimizer: {}\nloss: {}\n'.format(optimizer_name, loss_name))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        ModelCheckpoint(filepath=log_dir + 'ckpt_{}{}.h5'.format(model_name, model_version),
                        monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1),
        TensorBoard(log_dir=log_dir + 'tensorboard/', batch_size=batch_size),
        ]
    print('【training】')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  verbose=1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(loss))
    plt.figure()
    plt.plot(epochs_range, loss, 'b-', label='Training loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation loss')
    plt.ylim(ymin=0)
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(log_dir + 'loss_{}{}.png'.format(model_name, model_version))

    print('【prediction】')
    pred = model.predict_generator(test_gen,
                                   steps=test_steps,
                                   verbose=1)
    print('【evaluation】')
    eval = model.evaluate_generator(test_gen,
                                    steps=test_steps,
                                    verbose=1)
    naive_eval = naive_evaluate(test_gen, test_steps, pre_mean_loss, target_length, naive_period)

    pred_days = 92
    pred = pred[-pred_days:, :]
    naive_pred = float_data[-(2*pred_days + naive_period): -(pred_days + naive_period), -259:]
    true = float_data[-pred_days:, -259:]
    df_pred = pd.DataFrame(pred.sum(axis=0), index=latlon)
    df_naive_pred = pd.DataFrame(naive_pred.sum(axis=0), index=latlon)
    df_true = pd.DataFrame(true.sum(axis=0), index=latlon)
    df_eval = pd.concat([df_naive_pred, df_pred, df_true], axis=1)
    df_eval.columns = ['na_pred', 'pred', 'true']
    df_eval['na_eval'] = pre_mean_loss(df_eval['true'], df_eval['na_pred'])
    df_eval['eval'] = pre_mean_loss(df_eval['true'], df_eval['pred'])
    col_list = ['true', 'pred', 'eval', 'na_pred', 'na_eval']
    df_eval = df_eval.loc[:, col_list]
    sr_sum = pd.Series(df_eval.sum(axis=0), index=col_list, name='sum')
    sr_max = pd.Series(df_eval.max(axis=0), index=col_list, name='max')
    sr_mean = pd.Series(df_eval.mean(axis=0), index=col_list, name='mean')
    print('【result (last 3month)】')
    print(df_eval.sort_values(by='eval').append(sr_sum).append(sr_max).append(sr_mean).tail(20))
    df_eval = df_eval.append(sr_sum).append(sr_max).append(sr_mean)
    df_eval.to_csv(log_dir + 'eval_{}{}.csv'.format(model_name, model_version))
    print('【result (all)】\n model: {}\n naive: {}'.format(eval, naive_eval))

    if not record:
        return
    now = datetime.datetime.now().strftime(date_format)
    record_file = log_dir + 'record.csv'
    str_num_layers = list_to_str(num_layers)
    str_dropouts = list_to_str(dropouts)
    str_recurrent_dropouts = list_to_str(recurrent_dropouts)
    if os.path.exists(record_file):
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('{},{},{}{},{},{},{},{},{},{},{}\n'
                    .format(now, eval, model_name, model_version, str_num_layers, optimizer_name,
                            str_dropouts, str_recurrent_dropouts, epochs, batch_size, stateful))
    else:
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('date,score,model,num_layers,optimizer,dropouts,recurrent_dropouts,epochs,batch_size,stateful\n')
            f.write('{},{},{}{},{},{},{},{},{},{},{}\n'
                    .format(now, eval, model_name, model_version, str_num_layers, optimizer_name,
                            str_dropouts, str_recurrent_dropouts, epochs, batch_size, stateful))

if __name__ == '__main__':
    main()
