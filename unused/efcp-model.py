# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train, run_val
from modules.score import kaggle_score
from modules.utils import pad0
import os, torch
from datetime import datetime
import pandas as pd

run_name = 'files5-droput5-epochs1'

# train models. 
nets = {}
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    if True:
    # if not os.path.exists(f'out/train/train/net-{protein_name}.pt'):
        ids, net, labels, scores = train(
            get_loader('out/train/train/', protein_name, n_files = 5),
            epochs = 1,
            print_batches = 5000,
            save_folder = 'out/train/train/',
            save_name = f'net-{protein_name}'
        )
        nets[protein_name] = net
        del net, labels, scores
    else:
        nets[protein_name] = torch.jit.load(f'out/train/train/net-{protein_name}.pt').eval()

# check val to get expected score.
solution = []
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, labels, scores = run_val(get_loader('out/train/val/', protein_name, n_files = 5), nets[protein_name])
    
    solution.append(pd.DataFrame({
        'id': ids,
        'protein_name': [protein_name]*len(labels), 
        'binds': labels,
        'split_group': [1]*len(labels)
    }))
    submission.append(pd.DataFrame({
        'id': ids,
        'binds': scores
    }))
    
    del protein_name, ids, labels, scores

solution = pd.concat(solution)
submission = pd.concat(submission)
expected_score = kaggle_score(solution, submission, "id")
print(f'expected score: {expected_score :.2f}')

# run test to get the actual submission. 
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, labels, scores = run_val(get_loader('out/test/test/', protein_name), nets[protein_name])    
    
    submission.append(pd.DataFrame({
        'id': ids,
        'binds': scores
    }))
    
    del protein_name, ids, labels, scores
    
submission = pd.concat(submission).sort_values('id')
# submission.to_parquet(f'out/test/submission-{datetime.today().strftime("%Y%m%d")}-{pad0(int(expected_score*100))}.parquet', index = False)
submission.to_parquet(f'out/test/submission-{datetime.today().strftime("%Y%m%d")}-{run_name}-{pad0(int(expected_score*100))}.parquet', index = False)

