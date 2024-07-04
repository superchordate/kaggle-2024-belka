from modules.datasets import get_loader
from modules.train import run_val, get_model_optimizer
from modules.utils import gcp, dircreate, kaggle, cloud
from modules.score import kaggle_score, print_results
from datetime import datetime
import pandas as pd
import polars as pl
import torch, os
import numpy as np

options = {
    'lr': 0.001,
    'momentum': 0.9,
    'dropout': 0,
    'network': 'md'
}

run_name = 'md-500K-1e-reb30f-drop50-pca90-es4'

mols = pl.read_parquet('out/train/val/mols.parquet').sample(500*1000)
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
    net, options
)

train_val_distinct_blocks = pl.read_parquet('out/train/val/train_val_distinct_blocks.parquet')
mols_with_nonshared = mols.filter(
    pl.col('buildingblock1_index').is_in(train_val_distinct_blocks['index']) |
    pl.col('buildingblock2_index').is_in(train_val_distinct_blocks['index']) |
    pl.col('buildingblock3_index').is_in(train_val_distinct_blocks['index'])
)['molecule_id']

val_data = []
for protein_name in ['sEH', 'BRD4', 'HSA']:   
    val_data.append(pd.DataFrame({
        'id': molecule_ids,
        'protein_name': [protein_name]*len(molecule_ids), 
        'binds': labels[protein_name],
        'binds_predict': scores[protein_name],
        'split_group': np.where(np.isin(molecule_ids, mols_with_nonshared), 'nonshared', 'shared')
    }))
    
val_data = pd.concat(val_data).sort_values('id')
print_results(val_data['binds'], val_data['binds_predict'])

solution = val_data[['id', 'protein_name', 'binds', 'split_group']]
submission = val_data[['id', 'binds_predict']].rename({'binds_predict': 'binds'})
expected_score = kaggle_score(solution, submission, "id")
print(f'expected score: {expected_score :.3f}')


