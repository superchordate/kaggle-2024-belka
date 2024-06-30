from modules.train import train
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
    'print_batches': 2000,
    'network': 'lg'
}

run_name = 'md-allrows-3e'

model_path = f'{run_name}.pt'

os.system('gsutil cp gs://kaggle-417721/blocks-3-pca.parquet blocks-3-pca.parquet')
os.system('gsutil cp gs://kaggle-417721/mols.parquet mols.parquet')
os.system(f'gsutil cp gs://kaggle-417721/{model_path} {model_path}')

if not os.path.exists(model_path):
    net, labels, scores = train(
        indir = '.',
        options = options,
        print_batches = options['print_batches'],
        save_folder = '.',
        save_name = run_name
    )
elif useprior:
    net, labels, scores = train(
        indir = '.',
        options = options,
        print_batches = options['print_batches'],
        save_folder = '.',
        net = torch.jit.load(model_path).train(),
        save_name = run_name
    )
else:
    raise Exception('Model already exists')