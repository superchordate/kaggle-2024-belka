from modules.datasets import get_loader
from modules.train import run_val, get_model_optimizer
from modules.utils import gcp, dircreate, kaggle, cloud
from datetime import datetime
import pandas as pd
import polars as pl
import torch, os
import numpy as np

options = {
    'lr': 0.001,
    'momentum': 0.9,
    'dropout': 0,
    'network': 'lg'
}

run_name = 'md-500K-1e-reb30f-drop50-pca90-es4'

mols = pl.read_parquet('out/train/val/mols.parquet')
blocks = pl.read_parquet('out/blocks-3-pca.parquet')

datafolder = 'out/train/val/'
load_path = f'out/net/{run_name}'

net, optimizer = get_model_optimizer(
    options, mols, blocks, 
    load_path = load_path, 
    load_optimizer = False
)

molecule_ids, labels, scores = run_val(
    get_loader(datafolder, mols = mols, blocks = blocks, options = options), 
    net
)

val_data = []
for protein_name in ['sEH', 'BRD4', 'HSA']:   
    val_data.append(pd.DataFrame({
        'id': molecule_ids,
        'protein_name': [protein_name]*len(iy), 
        'binds': labels[protein_name],
        'binds_predict': scores[protein_name]
    }))
    
val_data = pd.concat(val_data).sort_values('id')
print_results(val_data['binds'], val_data['binds_predict'])

solution = val_data[['id', 'protein_name', 'binds', 'split_group']]
solution['split_group'] = np.random.choice(range(10), solution.shape[0], replace = True)
submission = val_data[['id', 'binds_predict']].rename({'binds_predict': 'binds'})
expected_score = kaggle_score(solution, submission, "id")
print(f'expected score: {expected_score :.3f}')


