# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train

run_name = 'files5-dropout50-epochs1-onehot'

for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, net, labels, scores = train(
        get_loader('out/train/train', protein_name, n_files = 5),
        epochs = 1,
        save_folder = 'out/net/',
        save_name = f'net-{run_name}-{protein_name}',
        print_batches = 2500
    )