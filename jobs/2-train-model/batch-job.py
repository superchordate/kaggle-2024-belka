from modules.train import train
from modules.utils import gcp
import os, torch

useprior = True

options = {
    'epochs': 3,
    'train_batch_size': 100,
    'lr': 0.001,
    'momentum': 0.9,
    'dropout': 50,
    'rebalanceto': 0.1,
    'n_rows': 'all',
    'print_batches': 2500,
    'network': 'md',
    'num_splits': 50 if gcp() else 125
}

run_name = 'md-allrows-3e'

if gcp():
    os.system('gsutil cp gs://kaggle-417721/blocks-3-pca.parquet blocks-3-pca.parquet')
    os.system('gsutil cp gs://kaggle-417721/mols.parquet mols.parquet')
    os.system(f'gsutil cp gs://kaggle-417721/{run_name}.state {run_name}.state')
    os.system(f'gsutil cp gs://kaggle-417721/{run_name}-opt.state {run_name}-opt.state')
    indir = '.' 
    save_folder = '.'
else:
    indir = 'out/train/'
    save_folder = 'out/net/'

if not os.path.exists(f'{save_folder}/{run_name}.state'):

    net, labels, scores = train(
        indir = indir,
        options = options,
        print_batches = options['print_batches'],
        save_folder = save_folder,
        save_name = run_name
    )

elif useprior:

    net, labels, scores = train(
        indir = indir,
        options = options,
        print_batches = options['print_batches'],
        save_folder = save_folder,
        model_load_path = f'{save_folder}/{run_name}',
        save_name = run_name
    )

else:
    raise Exception('Model already exists')