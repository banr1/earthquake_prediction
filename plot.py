import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import seaborn as sns
sns.set_style('white')
from mpl_toolkits.basemap import Basemap
import matplotlib.cbook as cbook
import warnings
warnings.filterwarnings('ignore', category=cbook.mplDeprecation)
from keras.models import load_model
from keras.utils import plot_model

from losses import *
from params import args

if __name__ == '__main__':
    mdl_name = args.model
    ver = args.version
    log_dir = args.log_dir
    fig_dir = '../figure/'

def save(filename):
    plt.savefig(filename, transparent=False, bbox_inches='tight')
    print('saved {}'.format(filename))

def plot_on_map(evals, cmax, filename):
    lat = np.arange(-89, 91)
    lon = np.arange(-179, 181)
    lon, lat = np.meshgrid(lon, lat)
    fig = plt.figure(figsize=(10, 8))
    m = Basemap(projection='lcc', resolution='l',
                width=2E6, height=2E6,
                lat_0=37.5, lon_0=137.5,)
    m.shadedrelief(scale=0.5)
    m.pcolormesh(lon, lat, evals,
                 latlon=True, cmap='jet')
    plt.clim(0, cmax)
    m.drawcoastlines(color='lightgray')
    plt.colorbar(label='Poisson Log Likelihood')
    save(fig_dir + filename)

def main():
    model = load_model(log_dir + 'ckpt_{}{}.h5'.format(mdl_name, ver),
                       custom_objects={'mean_poisson_log_likelihood':
                                       mean_poisson_log_likelihood})
    plot_model(model, show_shapes=True,
               to_file=fig_dir + 'mdl_{}{}.png'.format(mdl_name, ver))
    print('saved {}'.format(fig_dir + 'mdl_{}{}.png'.format(mdl_name, ver)))

    hist = pd.read_csv(log_dir + 'history_{}{}.csv'.format(mdl_name, ver))
    hist = hist[['epoch', 'loss', 'val_loss']].set_index('epoch')
    hist.columns = ['train loss', 'validation loss']
    hist.plot()
    save(fig_dir + 'loss_{}{}.png'.format(mdl_name, ver))

    eval_day = pd.read_csv(log_dir + 'eval_day_{}{}.csv'.format(mdl_name, ver),
                           index_col=0)
    eval_day.index = pd.to_datetime(eval_day.index)
    pred = eval_day.iloc[:, :3]
    pred.plot()
    save(fig_dir + 'pred_day_{}{}_Naive.png'.format(mdl_name, ver))
    eval = eval_day.iloc[:, 3:]
    eval.plot()
    save(fig_dir + 'eval_day_{}{}_Naive.png'.format(mdl_name, ver))

    eval_bin = pd.read_csv(log_dir + 'eval_bin_{}{}.csv'.format(mdl_name, ver),
                           index_col=0)
    pred = eval_bin.iloc[:, :3]
    pred.plot()
    save(fig_dir + 'pred_bin_{}{}_Naive.png'.format(mdl_name, ver))
    eval = eval_bin.iloc[:, 3:]
    eval.plot()
    save(fig_dir + 'eval_bin_{}{}_Naive.png'.format(mdl_name, ver))

    eval_bin = eval_bin.reset_index()
    eval_bin['lat'] = eval_bin['index'].astype(str).str[:2].astype(int)
    eval_bin['lon'] = eval_bin['index'].astype(str).str[3:].astype(int)
    eval_bin = eval_bin.iloc[:, 1:]
    trues, nv_preds, md_preds, nv_evals, md_evals = np.zeros((5, 180, 360))
    for idx, row in eval_bin.iterrows():
        la = row['lat'].astype(int)
        lo = row['lon'].astype(int)
        trues[la+89, lo+179] = row['True value']
        nv_preds[la+89, lo+179] = row['Naive predicton']
        md_preds[la+89, lo+179] = row['{} prediction'.format(mdl_name)]
        nv_evals[la+89, lo+179] = row['Naive error']
        md_evals[la+89, lo+179] = row['{} error'.format(mdl_name)]
    pred_cmax = np.max([trues, nv_preds, md_preds])
    eval_cmax = np.max([nv_evals, md_evals])
    plot_on_map(trues, pred_cmax, 'true.png')
    plot_on_map(nv_preds, pred_cmax, 'pred_bin_Naive.png')
    plot_on_map(md_preds, pred_cmax, 'pred_bin_{}{}.png'.format(mdl_name, ver))
    plot_on_map(nv_evals, eval_cmax, 'eval_bin_Naive.png')
    plot_on_map(md_evals, eval_cmax, 'eval_bin_{}{}.png'.format(mdl_name, ver))

if __name__ == '__main__':
    main()
