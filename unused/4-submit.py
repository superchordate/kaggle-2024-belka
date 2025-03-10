
from modules.datasets import get_loader
from modules.train import run_val
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
    'print_batches': 2000,
}

modelfile = 'smallnet-allrows-2e'

net = torch.jit.load(f'out/net/{modelfile}.pt').eval()

# run test to get the actual submission.
molecule_ids, labels, scores = run_val(
    get_loader('out/test/', options = options, submit = True), 
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

submission = pd.concat(submission).sort_values('id')

if submission.shape[0] != 1674896:
    raise Exception(f'Submission must have 1674896 rows, found: {submission.shape[0]}')

dircreate('out/submit/')
submitfile = f'out/submit/submission-{datetime.today().strftime("%Y%m%d")}-{modelfile}.parquet'
submission.to_parquet(submitfile, index = False)
