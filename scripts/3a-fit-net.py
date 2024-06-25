# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

from modules.datasets import get_loader
from modules.net import train, run_val, print_results
from modules.score import kaggle_score
from modules.utils import pad0, dircreate
import os, torch
from datetime import datetime
import pandas as pd

run_name = 'files5-dropout50-epochs3'
options = {
    'onehot': True, 
    'n_files': 5, 
    'epochs': 3,
    'print_batches': 5000,
}

#TODO - compare to 1 epcoch 15 files (same # batches/duration)

# train models. 
nets = {}   
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    model_path = f'out/net/net-{run_name}-{protein_name}.pt'
    if not os.path.exists(model_path):
        ids, net, labels, scores = train(
            get_loader('out/train/train/', protein_name, options = options),
            epochs = options['epochs'],
            print_batches = options['print_batches'],
            save_folder = 'out/net/',
            save_name = f'net-{run_name}-{protein_name}'
        )
        nets[protein_name] = net
        del net, labels, scores
    else:
        nets[protein_name] = torch.jit.load(model_path).eval()

# get metrics for train so we can see if we are over-fitting.
if 'n_files' in options:
    print('measure train accuracy for sEH')
    ids, labels, scores = run_val(
        get_loader('out/train/val/', 'sEH', options = options), 
        nets[protein_name]
    )
    print_results(labels, scores)

# check val to get expected score.
print('val')
solution = []
submission = []
for protein_name in ['sEH']:
#for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, labels, scores = run_val(
        get_loader('out/train/val/', protein_name, options = options), 
        nets[protein_name]
    )
    
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
print_results(solution['binds'], submission['binds'])

# run test to get the actual submission. 
print('submit')
submission = []
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    ids, labels, scores = run_val(
        get_loader('out/test/test/', protein_name, options = options, submit = True), 
        nets[protein_name]        
    )
    
    submission.append(pd.DataFrame({
        'id': ids,
        'binds': scores
    }))
    
    del protein_name, ids, labels, scores

dircreate('out/submit')
submission = pd.concat(submission).sort_values('id')
submission.to_parquet(f'out/submit/submission-{datetime.today().strftime("%Y%m%d")}-{run_name}-{pad0(int(expected_score*100))}.parquet', index = False)

