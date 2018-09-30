import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import datetime
import random
from tqdm import tqdm
import keras.optimizers
import keras.backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger

from params import args
import models, naives, losses

if __name__ == '__main__':
    mdl_name = args.model
    ver = args.version
    opt_name = args.optimizer
    learning_rate = args.learning_rate
    decay = args.decay
    ls_name = args.loss
    stateful = args.stateful
    lb = args.lookback
    batch_size = args.batch_size
    epochs = args.epochs
    num_filters = args.num_filters
    dropouts = args.dropouts
    recurrent_dropouts = args.recurrent_dropouts
    random_seed = args.random_seed
    naive_period = args.naive_period
    st_day = args.start_day
    sp1_day = args.split_day_1
    sp2_day = args.split_day_2
    ed_day = args.end_day
    input_raw_dir = args.input_raw_dir
    input_preprocessed_dir = args.input_preprocessed_dir
    log_dir = args.log_dir
    vb = args.verbose
    record = args.record
    fmt = '%Y-%m-%d'
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
    train_st = start_day
    train_ed = split_day_1 - datetime.timedelta(days=1)
    train_period = (train_ed - train_st).days + 1
    val_st = split_day_1
    val_ed = split_day_2 - datetime.timedelta(days=1)
    val_period = (val_ed - val_st).days + 1
    test_st = split_day_2
    test_ed = end_day
    test_period = (test_ed - test_st).days + 1
    print('train    period: {} ~ {} ({:0=4}days)'.format(train_st.strftime(fmt),
                                                         train_ed.strftime(fmt),
                                                         train_period))
    print('validate period: {} ~ {} ({:0=4}days)'.format(val_st.strftime(fmt),
                                                         val_ed.strftime(fmt),
                                                         val_period))
    print('test     period: {} ~ {} ({:0=4}days)'.format(test_st.strftime(fmt),
                                                         test_ed.strftime(fmt),
                                                         test_period))
    return train_period, val_period, test_period

def find_class(name, modules):
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
    df = df[start: end]
    idx = pd.DataFrame(index=pd.date_range(start, end))
    df = pd.merge(idx, df, how='outer', left_index=True, right_index=True)
    df = df.fillna(0)
    df = df.astype(int)
    return df

def get_test_true(test_gen, test_steps):
    for step in tqdm(range(test_steps)):
        _, target = next(test_gen)
        day_target = np.mean(target, axis=1)
        if step == 0:
            bin_targets = target
            day_targets = day_target
        else:
            bin_targets = np.vstack((bin_targets, target))
            day_targets = np.hstack((day_targets, day_target))
    return np.mean(bin_targets, axis=0), day_targets

def model_evaluate(test_gen, test_steps, pre_mean_loss, target_length, model):
    for step in tqdm(range(test_steps)):
        sample, target = next(test_gen)
        pred = model.predict(sample)
        day_pred = np.mean(pred, axis=1)
        bin_error = pre_mean_loss(target, pred)
        day_error = np.mean(bin_error, axis=1)
        if step == 0:
            bin_preds = pred
            day_preds = day_pred
            bin_errors = bin_error
            day_errors = day_error
        else:
            bin_preds = np.vstack((bin_preds, pred))
            day_preds = np.hstack((day_preds, day_pred))
            bin_errors = np.vstack((bin_errors, bin_error))
            day_errors = np.hstack((day_errors, day_error))
    return (np.mean(bin_preds, axis=0), day_preds,
            np.mean(bin_errors, axis=0), day_errors)

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

    train_period, val_period, test_period = get_period(st_day, sp1_day, sp2_day, ed_day)
    raw_files = sorted(glob.glob(input_raw_dir + 'h????'))
    csv_file = input_preprocessed_dir + 'df.csv'
    if not os.path.exists(csv_file):
        raw_to_csv(raw_files, csv_file)

    df = pd.read_csv(csv_file, low_memory=False)
    df_m2 = df[df['magnitude'] >= 20]
    df_m2 = get_grid_data(df_m2)
    latlon = np.sort(df_m2['latlon'].unique())
    df_m2 = get_daily_data(df_m2, start=st_day, end=ed_day, dummy_col=latlon)
    df_m4 = df[df['magnitude'] >= 40]
    df_m4 = get_grid_data(df_m4)
    df_m4 = get_daily_data(df_m4, start=st_day, end=ed_day, dummy_col=latlon)
    data_m2 = df_m2.values.astype(np.float64)
    data_m4 = df_m4.values.astype(np.float64)
    max_m2 = data_m2.max(axis=(0,1))
    max_m4 = data_m4.max(axis=(0,1))
    data_m2 = data_m2 * max_m4 / max_m2
    target_length = data_m4.shape[1]
    data = np.hstack([data_m2, data_m4])
    print('data shape: {}'.format(data.shape))

    lr = learning_rate if learning_rate else get_default_lr(opt_name)
    optimizer = find_class(opt_name, [keras.optimizers])(lr=lr, decay=decay)
    loss = find_class(ls_name, [losses, keras.losses])
    pre_mean_loss = find_class(ls_name.replace('mean_', ''), [losses])

    train_gen = generator(data,
                          lookback=lb,
                          min_idx=0,
                          max_idx=train_period - 1,
                          batch_size=batch_size,
                          target_length=target_length)
    val_gen = generator(data,
                        lookback=lb,
                        min_idx=train_period,
                        max_idx=train_period + val_period - 1,
                        batch_size=batch_size,
                        target_length=target_length)
    test_gen = generator(data,
                         lookback=lb,
                         min_idx=train_period + val_period,
                         max_idx=None,
                         batch_size=batch_size,
                         target_length=target_length)

    train_steps = (train_period - lb) // batch_size
    val_steps = (val_period - lb) // batch_size
    test_steps = (len(data) - (train_period + val_period) - lb) // batch_size
    naive_class = find_class('Poissonnaive', [naives])()
    model_class = find_class(mdl_name, [models])()
    naive = naive_class.build_naive(data,
                                    batch_size=batch_size,
                                    target_length=target_length)
    model = model_class.build_model(data,
                                    lookback=lb,
                                    batch_size=batch_size,
                                    optimizer=optimizer,
                                    loss=loss,
                                    stateful=stateful,
                                    target_length=target_length,
                                    num_filters=num_filters,
                                    dropouts=dropouts,
                                    recurrent_dropouts=recurrent_dropouts)
    model.summary()
    print('optimizer: {} (lr={})\nloss: {}\n'.format(opt_name, lr, ls_name))
    callbacks = [
        ModelCheckpoint(filepath=log_dir + 'ckpt_{}{}.h5'.format(mdl_name, ver),
                        monitor='val_loss', save_best_only=True, verbose=vb),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=10, verbose=vb),
        CSVLogger(log_dir + 'history_{}{}.csv'.format(mdl_name, ver)),
        ]
    print('【training】')
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=train_steps,
                                  epochs=epochs,
                                  validation_data=val_gen,
                                  validation_steps=val_steps,
                                  callbacks=callbacks,
                                  verbose=vb)

    print('【evaluation】')
    bin_true, day_true = get_test_true(test_gen, test_steps)
    true = np.mean(bin_true)
    nv_bin_pred, nv_day_pred, nv_bin_eval, nv_day_eval = model_evaluate(
            test_gen, test_steps, pre_mean_loss, target_length, naive)
    nv_pred = np.mean(nv_bin_pred)
    nv_eval = np.mean(nv_bin_eval)
    md_bin_pred, md_day_pred, md_bin_eval, md_day_eval = model_evaluate(
            test_gen, test_steps, pre_mean_loss, target_length, model)
    md_pred = np.mean(md_bin_pred)
    md_eval = np.mean(md_bin_eval)
    print('Naivemodel: {}'.format(nv_eval))
    print('{}{}: {}'.format(mdl_name, ver, md_eval))

    df_day_true = pd.DataFrame(day_true,
                               index=pd.date_range(end=ed_day, periods=93),
                               columns=['True value'])
    df_nv_day_pred = pd.DataFrame(nv_day_pred,
                                  index=pd.date_range(end=ed_day, periods=93),
                                  columns=['Naive predicton'])
    df_md_day_pred = pd.DataFrame(md_day_pred,
                                  index=pd.date_range(end=ed_day, periods=93),
                                  columns=['{} prediction'.format(mdl_name)])
    df_nv_day_eval = pd.DataFrame(nv_day_eval,
                                  index=pd.date_range(end=ed_day, periods=93),
                                  columns=['Naive error'])
    df_md_day_eval = pd.DataFrame(md_day_eval,
                                  index=pd.date_range(end=ed_day, periods=93),
                                  columns=['{} error'.format(mdl_name)])
    df_day_eval = pd.concat([df_nv_day_pred, df_md_day_pred, df_day_true,
                             df_nv_day_eval, df_md_day_eval],
                            axis=1)
    df_day_eval.to_csv(log_dir + 'eval_day_{}{}.csv'.format(mdl_name, ver))

    df_bin_true = pd.DataFrame(bin_true,
                               index=latlon,
                               columns=['True value'])
    df_nv_bin_pred = pd.DataFrame(nv_bin_pred,
                                  index=latlon,
                                  columns=['Naive predicton'])
    df_md_bin_pred = pd.DataFrame(md_bin_pred,
                                  index=latlon,
                                  columns=['{} prediction'.format(mdl_name)])
    df_nv_bin_eval = pd.DataFrame(nv_bin_eval,
                                  index=latlon,
                                  columns=['Naive error'])
    df_md_bin_eval = pd.DataFrame(md_bin_eval,
                                  index=latlon,
                                  columns=['{} error'.format(mdl_name)])
    df_bin_eval = pd.concat([df_nv_bin_pred, df_md_bin_pred, df_bin_true,
                             df_nv_bin_eval, df_md_bin_eval],
                            axis=1)
    df_bin_eval.to_csv(log_dir + 'eval_bin_{}{}.csv'.format(mdl_name, ver))

    if not record:
        return
    now = datetime.datetime.now().strftime(fmt)
    record_file = log_dir + 'record.csv'
    str_num_filters = list_to_str(num_filters)
    str_dropouts = list_to_str(dropouts)
    str_recurrent_dropouts = list_to_str(recurrent_dropouts)
    if os.path.exists(record_file):
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('{},{},{}{},{},{},{},{},{},{},{},{}\n'
                    .format(now, md_eval, mdl_name, ver, str_num_filters,
                            opt_name, lr, decay, str_dropouts,
                            str_recurrent_dropouts, epochs,random_seed))
    else:
        with open(log_dir + 'record.csv', 'a') as f:
            f.write('date,eval,model,filt,optm,lr,decay,drpout,r_drpout,epch,seed\n')
            f.write('{},{},Naivemodel,None,None,None,None,None,None,None,None\n'.format(now, nv_eval))
            f.write('{},{},{}{},{},{},{},{},{},{},{},{}\n'
                    .format(now, md_eval, mdl_name, ver, str_num_filters,
                            opt_name, lr, decay, str_dropouts,
                            str_recurrent_dropouts, epochs,random_seed))

if __name__ == '__main__':
    main()
