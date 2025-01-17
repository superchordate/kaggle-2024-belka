import lightgbm as lgb
import numpy as np
import polars as pl
import pandas as pd
from modules.utils import listfiles, pad0, load1
from modules.score import kaggle_score
from modules.mols import add_block_ecfps
import os
from datetime import datetime
from distributed import Client, LocalCluster

cluster = LocalCluster(n_workers=1)
client = Client(cluster)
dask_model = lgb.DaskLGBMClassifier(client=client)
dask_model.set_params(client=client)

n_files = None
run_name = 'filesall'

params = {
    "objective": "binary",
    "max_depth": 15,
    "num_leaves": 32,
    "min_data_in_leaf": 1000,
    "learning_rate": 0.01,
    "verbose": -1,
    # 'metric': ['binary'],
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'is_unbalance': True,
    'bagging_fraction': 0.5,
    'feature_fraction': 0.5,
    'feature_fraction_bynode': 0.5,
    'extra_trees': True,
    'max_bin': 100
}

train_blocks = load1('out/train/train_blocks.pkl')
val_blocks = load1('out/train/val_blocks.pkl')
# train_blocks_all = pl.read_parquet('out/train/building_blocks.parquet').to_pandas()
test_blocks = pl.read_parquet('out/test/building_blocks.parquet').to_pandas()

gbms = {}
val_data = []
blocks = pl.read_parquet('out/train/building_blocks.parquet')
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    model_path = f'out/gbm/gbm-{run_name}-{protein_name}.gbm'

    if not os.path.exists(model_path):

        print(f'Training model for {protein_name}')

        print('train model')
        ct = 0
        dofiles = listfiles('out/train/train/base/', protein_name)
        if n_files: 
            dofiles = np.random.choice(dofiles, n_files)
        else:
            n_files = len(dofiles)
        for file in dofiles:
            ct+= 1
            print(f'    {protein_name} {ct} of {n_files}: {file}')
            idt = add_block_ecfps(file, train_blocks).select(['binds', 'ecfp_pca'])
            iX = np.vstack(idt['ecfp_pca'].to_numpy())
            iy = idt['binds'].to_numpy()
            
            lgb_train = lgb.Dataset(iX, iy)
            if 'gbm' not in globals():
                gbm = lgb.train(params, lgb_train, num_boost_round=10)
            else:
                gbm = lgb.train(params, lgb_train, num_boost_round=10, init_model=gbm)
            
            gbm.save_model(model_path)
            del file, idt, iX, iy, lgb_train

        gbms[protein_name] = gbm
        del gbm
        

print('validate model')
val_data = []
for protein_name in ['sEH', 'BRD4', 'HSA']:
        
    model_path = f'out/gbm/gbm-{run_name}-{protein_name}.gbm'
    gbms[protein_name] = lgb.Booster(model_file = model_path)
    
    dofiles = listfiles('out/train/val/base/', protein_name)
    if n_files: dofiles = np.random.choice(dofiles, n_files)

    for file in dofiles:
        
        print(f'    {file}')
        idt = add_block_ecfps(file, val_blocks).select(['binds', 'ecfp_pca'])
        iX = np.vstack(idt['ecfp_pca'].to_numpy())
        iy = idt['binds'].to_numpy()

        val_data.append(pd.DataFrame({
            'id': pl.read_parquet(file)['id'],
            'protein_name': [protein_name]*len(iy), 
            'binds': iy,
            'binds_predict': gbms[protein_name].predict(iX),
            'split_group': [1]*len(iy)
        }))
        
        del file, idt, iX, iy


val_data = pd.concat(val_data).sort_values('id')
solution = val_data[['id', 'protein_name', 'binds', 'split_group']]
submission = val_data[['id', 'binds_predict']].rename({'binds_predict': 'binds'})
expected_score = kaggle_score(solution, submission, "id")
print(f'expected score: {expected_score :.2f}')

# run test to get the actual submission.
print('submit')
submission = []
blocks = pl.read_parquet('out/test/building_blocks.parquet')
for protein_name in ['sEH', 'BRD4', 'HSA']:
        
    dt = []
    for file in listfiles('out/test/test/base/', protein_name):
        dt.append(add_block_ecfps(file, test_blocks))
        del file
    dt = pl.concat(dt).select(['id', 'ecfp_pca'])

    scores_test = gbms[protein_name].predict(
        np.vstack(dt['ecfp_pca'].to_numpy()), 
        num_iteration=gbms[protein_name].best_iteration
    )

    submission.append(pd.DataFrame({
        'id': dt['id'].to_numpy(),
        'binds': scores_test
    }))
    
    del protein_name, scores_test, dt
    
submission = pd.concat(submission).sort_values('id')
submission.to_parquet(f'out/gbm/submission-{datetime.today().strftime("%Y%m%d")}-lgbm-{run_name}-{pad0(int(expected_score*100))}.parquet', index = False)



