# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train, run_val
from modules.score import kaggle_score, print_results
from modules.utils import pad0, dircreate
import os, torch
from datetime import datetime
import pandas as pd
import polars as pl
import numpy as np

#dircreate('out/net', fromscratch = True)

useprior = False

options = {
    'epochs': 5,
    'train_batch_size': 100,
    'lr': 0.001,
    'momentum': 0.9,
    'onehot': True,
    'ecfp': True,
    'dropout': 50,
    'n_rows': 4*1000*1000,
    'print_batches': 1000,
}

#run_name = f'epochs{options["epochs"]}-trainbatch{options["train_batch_size"]}-dropout{options["dropout"]}-n_rows{options["n_rows"]}'
run_name = 'latest'

# train model
model_path = f'out/net/{run_name}.pt'
if not os.path.exists(model_path):
    net, labels, scores = train(
        get_loader(indir = 'out/train/train/', options = options),
        options = options,
        print_batches = options['print_batches'],
        save_folder = 'out/net/',
        save_name = f'{run_name}'
    )
elif useprior:
    net, labels, scores = train(
        get_loader(indir = 'out/train/train/', options = options),
        options = options,
        print_batches = options['print_batches'],
        save_folder = 'out/net/',
        net = torch.jit.load(model_path).train(),
        save_name = f'{run_name}'
    )    
else:
    net = torch.jit.load(model_path).eval()

# get metrics for train so we can see if we are over-fitting.
print('measure train accuracy')
molecule_ids, labels, scores = run_val(
    get_loader(indir = 'out/train/train/', options = options, checktrain = True), 
    net
)
for protein_name in ['sEH', 'BRD4', 'HSA']:
    print_results(labels[protein_name], scores[protein_name], metrics = ['average_precision_score', 'gini'])

# check val to get expected score.
print('val')
molecule_ids, labels, scores = run_val(
    get_loader('out/train/val/', options = options, checktrain = True), 
    net
)
for protein_name in ['sEH', 'BRD4', 'HSA']:
    print_results(labels[protein_name], scores[protein_name], metrics = ['average_precision_score', 'gini'])

# build out the solution and submission (val).
solution = []
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:

    results = pl.from_pandas(pd.DataFrame({
        'molecule_id': molecule_ids,
        'binds_predict': scores[protein_name],
        'binds_actual': labels[protein_name]
    })).with_columns(pl.col('molecule_id').cast(pl.Float32))
    
    inputs = pl.read_parquet(f'out/train/train-{protein_name}-idsonly.parquet').with_columns(pl.col('molecule_id').cast(pl.Float32))
    inputs = inputs.join(results, on = 'molecule_id', how = 'inner').drop('molecule_id')

    isolution = inputs.select(['id', 'binds_actual']).rename({'binds_actual': 'binds'})
    isolution = isolution.with_columns(pl.Series('protein_name', [protein_name]*isolution.shape[0]))
    isolution = isolution.with_columns(pl.Series('split_group', np.random.choice(range(25), isolution.shape[0])))
    isolution = isolution.select(['id', 'protein_name', 'binds', 'split_group'])

    isubmission = inputs.select(['id', 'binds_predict']).rename({'binds_predict': 'binds'})
    isubmission = isubmission.select(['id', 'binds'])

    solution.append(isolution.to_pandas())
    submission.append(isubmission.to_pandas())

    del isolution, isubmission, inputs, results, protein_name

solution = pd.concat(solution)
submission = pd.concat(submission)

expected_score = kaggle_score(solution, submission, "id")
print(f'expected score: {expected_score :.2f}')

del molecule_ids, labels, scores

# run test to get the actual submission. 
print('submit')
molecule_ids, labels, scores = run_val(
    get_loader('out/test/test/', options = options, submit = True), 
    net
)
del labels
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:

    results = pl.from_pandas(pd.DataFrame({
        'molecule_id': molecule_ids,
        'binds': scores[protein_name]
    })).with_columns(pl.col('molecule_id').cast(pl.Float32))
    
    inputs = pl.read_parquet(f'out/test/test-{protein_name}-idsonly.parquet').with_columns(pl.col('molecule_id').cast(pl.Float32))
    inputs = inputs.join(results, on = 'molecule_id', how = 'left')
    if inputs.null_count()['binds'][0] > 0:
        raise Exception(f'{inputs.null_count()["binds"][0]:,.0f} nulls after join.')        
    
    inputs = inputs .drop('molecule_id')

    isubmission = inputs.select(['id', 'binds'])
    isubmission = isubmission.select(['id', 'binds'])

    submission.append(isubmission.to_pandas())

    del isubmission, inputs, results, protein_name

dircreate('out/submit')
submission = pd.concat(submission).sort_values('id')

if submission.shape[0] != 1674896:
    raise Exception(f'Submission must have 1674896 rows, found: {submission.shape[0]}')

submission.to_parquet(
    f'out/submit/submission-{datetime.today().strftime("%Y%m%d")}-{run_name}-{pad0(int(expected_score*100))}.parquet', 
    index = False
)

