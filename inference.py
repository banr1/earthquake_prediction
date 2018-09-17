import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import datetime
import random
from PIL import Image
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
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
    learning_rate = args.learning_rate
    decay = args.decay
    loss_name = args.loss
    stateful = args.stateful
    lookback = args.lookback
    batch_size = args.batch_size
    epochs = args.epochs
    num_filters = args.num_filters
    dropouts = args.dropouts
    recurrent_dropouts = args.recurrent_dropouts
    random_seed = args.random_seed
    tensorboard = args.tensorboard
    naive_period = args.naive_period
    start_day = args.start_day
    split_day_1 = args.split_day_1
    split_day_2 = args.split_day_2
    end_day = args.end_day
    input_raw_dir = args.input_raw_dir
    input_preprocessed_dir = args.input_preprocessed_dir
    log_dir = args.log_dir
    verbose = args.verbose
    record = args.record
    date_format = '%Y-%m-%d'
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(random_seed)
    random.seed(random_seed)
    tf.set_random_seed(random_seed)

def list_to_str(list):
    return ' '.join(map(str, list))

def get_default_lr(optimizer_name):
    dict = {'SGD': 0.01, 'RMSprop':0.001, 'Adagrad': 0.01, 'Adadelta':1.0, 'Adam': 0.001, 'Nadam': 0.002}
    return dict[optimizer_name]

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
    df['latlon'] = df['latitude'].astype(str) + '-' + df['longitude'].astype(str)
    df_area = pd.read_table(input_raw_dir + 'mainland.forecast.nodes.dat',
                             names=('lon', 'lat'),
                             delim_whitespace=True,)
    df_area['latlon'] = df_area['lat'].astype(str).str[:2] + '-' + df_area['lon'].astype(str).str[:3]
    area = list(df_area['latlon'].unique())
    df = df[df['latlon'].isin(area)]
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
    for step in tqdm(range(test_steps)):
        samples, targets = next(test_gen)
        preds = samples[:, -naive_period, -target_length:]
        bin_error = pre_mean_loss(targets, preds)
        day_error = np.mean(bin_error, axis=1)
        if step == 0:
            bin_errors = bin_error
            day_errors = day_error
        else:
            bin_errors = np.vstack((bin_errors, bin_error))
            day_errors = np.hstack((day_errors, day_error))
    return np.mean(bin_errors, axis=0), day_errors, np.mean(day_errors)

def pred_evaluate(test_gen, test_steps, pre_mean_loss, target_length, model):
    errors = []
    for step in tqdm(range(test_steps)):
        samples, targets = next(test_gen)
        preds = model.predict(samples)
        bin_error = pre_mean_loss(targets, preds)
        day_error = np.mean(bin_error, axis=1)
        if step == 0:
            bin_errors = bin_error
            day_errors = day_error
        else:
            bin_errors = np.vstack((bin_errors, bin_error))
            day_errors = np.hstack((day_errors, day_error))
    return np.mean(bin_errors, axis=0), day_errors, np.mean(day_errors)

def generator(data, lookback, min_idx, max_idx, batch_size, target_length):
    if max_idx is None:
        max_idx = len(data) - 1
    i = min_idx + lookback
    while 1:
        if i + batch_size > max_idx + 1:
            i = min_idx + lookback
        rows = np.arange(i, min(i + batch_size, max_idx + 1))
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
    plt.savefig(log_dir + 'fig_train_hist.png')
    target_length = float_data_m4.shape[1]
    float_data = np.hstack([float_data_m2, float_data_m4])
    print('data shape: {}'.format(float_data.shape))

    lr = learning_rate if learning_rate else get_default_lr(optimizer_name)
    optimizer = find_class_by_name(optimizer_name, [keras.optimizers])(lr=lr, decay=decay)
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
                                    num_filters=num_filters,
                                    dropouts=dropouts,
                                    recurrent_dropouts=recurrent_dropouts)
    model.summary()
    print('optimizer: {} (lr={})\nloss: {}\n'.format(optimizer_name, lr, loss_name))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=verbose),
        ModelCheckpoint(filepath=log_dir + 'ckpt_{}{}.h5'.format(model_name, model_version),
                        monitor='val_loss', save_best_only=True, verbose=verbose),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=verbose),
        ]
    if tensorboard:
        callbacks.append(TensorBoard(log_dir=log_dir + 'tensorboard/', batch_size=batch_size))
    print('【training】')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  verbose=verbose)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(loss))
    plt.figure()
    plt.plot(epochs_range, loss, 'b-', label='Training loss')
    plt.plot(epochs_range, val_loss, 'r-', label='Validation loss')
    plt.ylim(ymin=0)
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(log_dir + 'fig_{}{}_loss.png'.format(model_name, model_version))

    print('【evaluation】')
    nv_bin_eval, nv_day_eval, nv_eval = naive_evaluate(test_gen, test_steps, pre_mean_loss, target_length, naive_period)
    md_bin_eval, md_day_eval, md_eval = pred_evaluate(test_gen, test_steps, pre_mean_loss, target_length, model)
    print('Naivemodel: {}'.format(nv_eval))
    print('{}{}: {}'.format(model_name, model_version, md_eval))

    df_nv_day_eval = pd.DataFrame(nv_day_eval, index=pd.date_range(end=end_day, periods=93), columns=['Naivemodel'])
    df_md_day_eval = pd.DataFrame(md_day_eval, index=pd.date_range(end=end_day, periods=93), columns=[model_name])
    df_day_eval = pd.concat([df_md_day_eval, df_nv_day_eval], axis=1)
    df_day_eval.plot()
    plt.savefig(log_dir + 'fig_{}{}_eval_day_vs_nv.png'.format(model_name, model_version))

    df_nv_bin_eval = pd.DataFrame(nv_bin_eval, index=latlon, columns=['Naivemodel'])
    df_md_bin_eval = pd.DataFrame(md_bin_eval, index=latlon, columns=[model_name])
    df_bin_eval = pd.concat([df_md_bin_eval, df_nv_bin_eval], axis=1)
    df_bin_eval.plot()
    plt.savefig(log_dir + 'fig_{}{}_eval_bin_vs_nv.png'.format(model_name, model_version))

    df_bin_eval = df_bin_eval.reset_index()
    df_bin_eval['lat'] = df_bin_eval['index'].astype(str).str[:2].astype(int)
    df_bin_eval['lon'] = df_bin_eval['index'].astype(str).str[3:].astype(int)
    df_bin_eval = df_bin_eval.iloc[:, 1:]

    lat = np.arange(-89, 91)
    lon = np.arange(-179, 181)
    lon, lat = np.meshgrid(lon, lat)
    nv_bin_evals = np.zeros((180, 360))
    md_bin_evals = np.zeros((180, 360))

    for idx, row in df_bin_eval.iterrows():
        la = row['lat'].astype(int)
        lo = row['lon'].astype(int)
        nv_bin_evals[la+89, lo+179] = row['Naivemodel']
        md_bin_evals[la+89, lo+179] = row[model_name]

    fig = plt.figure(figsize=(10, 8))
    m = Basemap(projection='lcc', resolution='l',
                width=2E6, height=2E6,
                lat_0=37.5, lon_0=137.5,)
    m.shadedrelief(scale=0.5)
    m.pcolormesh(lon, lat, nv_bin_evals,
                 latlon=True, cmap='jet')
    plt.clim(0, 2)
    m.drawcoastlines(color='lightgray')
    plt.title('Naivemodel')
    plt.colorbar(label='Poisson Log Likelihood')
    plt.savefig(log_dir + 'fig_{}_eval_bin.png'.format('Naivemodel'))

    fig = plt.figure(figsize=(10, 8))
    m = Basemap(projection='lcc', resolution='l',
                width=2E6, height=2E6,
                lat_0=37.5, lon_0=137.5,)
    m.shadedrelief(scale=0.5)
    m.pcolormesh(lon, lat, md_bin_evals,
                 latlon=True, cmap='jet')
    plt.clim(0, 2)
    m.drawcoastlines(color='lightgray')
    plt.title(model_name)
    plt.colorbar(label='Poisson Log Likelihood')
    plt.savefig(log_dir + 'fig_{}{}_eval_bin.png'.format(model_name, model_version))

    if not record:
        return
    now = datetime.datetime.now().strftime(date_format)
    record_file = log_dir + 'record.csv'
    str_num_filters = list_to_str(num_filters)
    str_dropouts = list_to_str(dropouts)
    str_recurrent_dropouts = list_to_str(recurrent_dropouts)
    if os.path.exists(record_file):
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('{},{},{}{},{},{},{},{},{},{},{},{}\n'
                    .format(now, md_eval, model_name, model_version, str_num_filters, optimizer_name, lr, decay,
                            str_dropouts, str_recurrent_dropouts, epochs,random_seed))
    else:
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('date,eval,model,filt,optm,lr,decay,drpout,r_drpout,epch,seed\n')
            f.write('{},{},Naivemodel,None,None,None,None,None,None,None,None\n'.format(now, nv_eval))
            f.write('{},{},{}{},{},{},{},{},{},{},{},{}\n'
                    .format(now, md_eval, model_name, model_version, str_num_filters, optimizer_name, lr, decay,
                            str_dropouts, str_recurrent_dropouts, epochs,random_seed))

if __name__ == '__main__':
    main()
