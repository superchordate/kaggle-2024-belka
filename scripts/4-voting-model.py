# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import run_val
from modules.utils import listfiles
from modules.mols import add_block_ecfps
import torch
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
from datetime import datetime

run_name = 'average'
print_batches = 100

net_name = 'filesall-droput5-epochs1'
gbm_name = 'filesall'

# load.
print('load models')
nets = {}
gbms = {}
for protein_name in ['sEH', 'BRD4', 'HSA']:    

    # net
    model_path = f'out/net/net-{net_name}-{protein_name}.pt'
    nets[protein_name] = torch.jit.load(model_path).eval()
    
    # gbm
    model_path = f'out/gbm/gbm-{gbm_name}-{protein_name}.gbm'
    gbms[protein_name] = lgb.Booster(model_file = model_path)

# net submission.
print('net submission:')
submission_net = []
for protein_name in ['sEH', 'BRD4', 'HSA']:    
    ids, labels, scores = run_val(get_loader('out/test/test/', protein_name), nets[protein_name])
    submission_net.append(pd.DataFrame({
        'id': ids,
        'binds': scores
    }))    
    del protein_name, ids, labels, scores
    
submission_net = pd.concat(submission_net).sort_values('id')

# gbm submission.
print('gbm submission:')
submission_gbm = []
test_blocks = pl.read_parquet('out/test/building_blocks.parquet').to_pandas()
for protein_name in ['sEH', 'BRD4', 'HSA']:
        
    dt = []
    for file in listfiles('out/test/test/base/', protein_name):
        dt.append(add_block_ecfps(file, test_blocks))
        del file
    dt = pl.concat(dt).select(['id', 'ecfp_pca'])

    X_test = np.vstack(dt['ecfp_pca'].to_numpy())
    id_test = dt['id'].to_numpy()
    scores_test = gbms[protein_name].predict(X_test, num_iteration=gbms[protein_name].best_iteration)

    submission_gbm.append(pd.DataFrame({
        'id': id_test,
        'binds': scores_test
    }))
    
    del protein_name, X_test, id_test, scores_test, dt
    
submission_gbm = pd.concat(submission_gbm).sort_values('id')

submission = pd.DataFrame({
    'id': submission_net['id'],
    'binds': (submission_net['binds'] + submission_gbm['binds'])/2
})

submission.to_parquet(f'out/voting/submission-{datetime.today().strftime("%Y%m%d")}-voting-{run_name}.parquet', index = False)

