# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train, run_val
from modules.score import kaggle_score
from modules.utils import pad0
import os, torch
from datetime import datetime
import pandas as pd

# download data.
# the container will have already downloaded modules. 
os.system(f'gsutil cp gs://kaggle-417721/base.zip base.zip && unzip base.zip && rm base.zip')
os.system(f'gsutil cp gs://kaggle-417721/building_blocks.parquet building_blocks.parquet')

run_name = 'files5-droput5-epochs3'

# for protein_name in ['sEH', 'BRD4', 'HSA']:
for protein_name in [['sEH', 'BRD4', 'HSA'][int(os.environ.get('BATCH_TASK_INDEX'))]]:
    
    ids, net, labels, scores = train(
        get_loader('', protein_name, n_files = 5),
        epochs = 3,
        save_folder = '',
        save_name = f'net-{run_name}-{protein_name}',
        print_batches = 10000
    )

