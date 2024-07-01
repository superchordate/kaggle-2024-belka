from modules.datasets import get_loader
from modules.train import run_val, get_model_optimizer
from modules.utils import gcp, dircreate
from datetime import datetime
import pandas as pd
import polars as pl
import torch, os

options = {
    'epochs': 3,
    'train_batch_size': 100,
    'lr': 0.001,
    'momentum': 0.9,
    'dropout': 50,
    'rebalanceto': 0.1,
    'n_rows': 'all',
    'print_batches': 1000,
    'network': 'md',
    'num_splits': 75 if gcp() else 300
}

run_name = 'md-allrows-3e'

if gcp():
    os.system(f'gsutil cp gs://kaggle-417721/{modelfile}.state {modelfile}.state')
    os.system(f'gsutil cp gs://kaggle-417721/blocks-3-pca.parquet blocks-3-pca.parquet')
    os.system('gsutil cp gs://kaggle-417721/test.zip test.zip')
    os.system('unzip test.zip')
    mols = pl.read_parquet('test/mols.parquet')
    blocks = pl.read_parquet('blocks-3-pca.parquet')
else:    
    mols = pl.read_parquet('out/test/mols.parquet')
    blocks = pl.read_parquet('out/blocks-3-pca.parquet')

net, optimizer = get_model_optimizer(
    options, mols, blocks, 
    load_path = run_name if gcp() else f'out/net/{run_name}', 
    load_optimizer = False
)

# run test to get the actual submission.
molecule_ids, labels, scores = run_val(
    get_loader('test/' if gcp() else 'out/test/', options = options, submit = True), 
    net
)
del labels
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:

    results = pl.from_pandas(pd.DataFrame({
        'molecule_id': molecule_ids,
        'binds': scores[protein_name]
    })).with_columns(pl.col('molecule_id').cast(pl.Float32))
    
    inputs = pl.read_parquet(
        f'test/test-{protein_name}-idsonly.parquet' if gcp() else f'out/test/test-{protein_name}-idsonly.parquet',
    ).with_columns(pl.col('molecule_id').cast(pl.Float32))
    inputs = inputs.join(results, on = 'molecule_id', how = 'left')
    if inputs.null_count()['binds'][0] > 0:
        raise Exception(f'{inputs.null_count()["binds"][0]:,.0f} nulls after join.')        
    
    inputs = inputs .drop('molecule_id')

    isubmission = inputs.select(['id', 'binds'])
    isubmission = isubmission.select(['id', 'binds'])

    submission.append(isubmission.to_pandas())

    del isubmission, inputs, results, protein_name

submission = pd.concat(submission).sort_values('id')

if submission.shape[0] != 1674896:
    raise Exception(f'Submission must have 1674896 rows, found: {submission.shape[0]}')

if gcp():
    submitfile = f'submission-{datetime.today().strftime("%Y%m%d")}-{run_name}.parquet'
    submission.to_parquet(submitfile, index = False)
    os.system(f'gsutil cp {submitfile} gs://kaggle-417721/{submitfile}')
else:
    dircreate('out/submit/')
    submitfile = f'out/submit/submission-{datetime.today().strftime("%Y%m%d")}-{run_name}.parquet'
    submission.to_parquet(submitfile, index = False)

