import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import keras.optimizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from params import args
import model_factory

if __name__ == '__main__':
    INPUT_RAW_DIR = '../input/raw/'
    INPUT_PREPROCESSED_DIR = '../input/preprocessed/'

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
    df['l/l'] = df['latitude'].astype(str) + '-' + df['longitude'].astype(str)
    df = df[['year', 'month', 'day', 'l/l']]
    df = df.set_index('l/l')
    df = df.reset_index()
    return df

def get_daily_data(df, start, end, dummy_col):
    df['time'] = df['year'].astype(str) + '-' + df['month'].astype(str) + '-' + df['day'].astype(str)
    df['time'] = pd.to_datetime(df['time'])
    df = df[['time', 'l/l']]
    df = pd.get_dummies(df, columns=['l/l'], prefix='', prefix_sep='')
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

def train_and_validate(float_data, dummy_col):
    lookback = args.lookback
    step = args.step
    delay = args.delay
    batch_size = args.batch_size
    optimizer_class = find_class_by_name(args.optimizer, [keras.optimizers])
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_idx=0,
                          max_idx=args.train_max_idx,
                          shuffle=False,
                          step=step,
                          batch_size=batch_size)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_idx=args.train_max_idx + 1,
                        max_idx=args.val_max_idx,
                        shuffle=False,
                        step=step,
                        batch_size=batch_size)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_idx=args.val_max_idx + 1,
                         max_idx=args.test_max_idx,
                         shuffle=False,
                         step=step,
                         batch_size=batch_size)
    train_steps = (4000 - 0 - lookback) // batch_size
    val_steps = (5000 - 4001 - lookback) // batch_size
    test_steps = (len(float_data) - 5001 - lookback) // batch_size
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
        #TensorBoard(log_dir='../log/', histogram_freq=1, embeddings_freq=1
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
    raw_files = glob.glob(INPUT_RAW_DIR + 'h20*')
    raw_files.sort()
    csv_file = INPUT_PREPROCESSED_DIR + 'df.csv'
    if not os.path.exists(csv_file):
        raw_to_csv(raw_files, csv_file)
    df = pd.read_csv(csv_file)
    df_m2 = df[df['magnitude'] >= 20]
    df_m2 = get_grid_data(df_m2)
    latlon = df_m2['l/l'].unique()
    df_m2 = get_daily_data(df_m2, start=args.start_day, end=args.end_day, dummy_col=latlon)
    df_m4 = df[df['magnitude'] >= 40]
    df_m4 = get_grid_data(df_m4)
    df_m4 = get_daily_data(df_m4, start=args.start_day, end=args.end_day, dummy_col=latlon)
    print(df_m2.head())
    print(df_m4.head())
    float_data_m2 = df_m2.values.astype(np.float64)
    float_data_m4 = df_m4.values.astype(np.float64)
    float_data_m2 = scaling(float_data_m2, 4000)
    float_data_m4 = scaling(float_data_m4, 4000)
    float_data = np.hstack([float_data_m2, float_data_m4])
    print(float_data.shape)
    train_and_validate(float_data, latlon)
    return

if __name__ == '__main__':
    main()

















""" CSEPに関するメモ """
#予測検証期間
# 1日, 3ヶ月, 1年, 3年
#予測する地震数のマグニチュード
# 3ヶ月以内: M>=4
# 1年以上: M>=5
#予測検証手法
# N-test (地震総数テスト): N
# M-test (地震規模の頻度分布テスト): γ
# L-test (時空間規模分布の尤度テスト): κ
# S-test (空間テスト): ζ
# R-test (尤度比テスト)

""" 現在未使用関数置き場 """
#全期間の場所ごとの地震数を可視化
def plot_all_eq(df):
    df = df.sum()
    df = df.reset_index()
    df['lat'] = df['index'].str[:2].astype(int)
    df['lon'] = df['index'].str[3:].astype(int)
    df = df.iloc[:, 1:]
    df.columns = ['num_seis', 'lat', 'lon']
    print(df)
