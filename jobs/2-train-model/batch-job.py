from modules.train import train
from modules.utils import gcp, kaggle
import os, torch

useprior = True

options = {
    'epochs': 2,
    'train_batch_size': 32,
    'lr': 0.005,
    'momentum': 0.9,
    'dropout': 50,
    'n_rows': 50*1000,
    'print_batches': 100,
    'network': 'sm',
    'num_splits': 3
    
}

run_name = 'test'
os.remove(f'out/net/{run_name}-champion.state')

# 1e done

# adam
# loss rate
# pca 95
# batch size

# baseline md 500K 1e rebalance0 dropout50 pca90 = 0.174
# rebalance10 dropout50 = 0.298
# rebalance25 dropout50 = 0.293
# rebalance10 dropout25 = 0.28
# rebalance10 dropout50 mom80 = 0.293
# rebalance10 dropout50 pca99 = 0.28
# rebalance10 dropout50 pca95 = 0.287
# rebalance10 dropout50 pca95 fixrebalance = 0.275

# strategy 
# lg model (3e has 0.43)
# rebalance10 
# dropout50
# pca90

if gcp():
    os.system('gsutil cp gs://kaggle-417721/blocks-3-pca.parquet blocks-3-pca.parquet')
    os.system('gsutil cp gs://kaggle-417721/mols.parquet mols.parquet')
    os.system(f'gsutil cp gs://kaggle-417721/{run_name}.state {run_name}.state')
    os.system(f'gsutil cp gs://kaggle-417721/{run_name}-opt.state {run_name}-opt.state')
    indir = '.' 
    save_folder = '.'
elif kaggle():
    indir = '.' 
    save_folder = '.'
else:
    indir = 'out/train/train/'
    save_folder = 'out/net/'

net, labels, scores = train(
    indir = indir,
    options = options,
    print_batches = options['print_batches'],
    save_folder = save_folder,
    model_load_path = f'{save_folder}/{run_name}-champion',
    save_name = run_name
)