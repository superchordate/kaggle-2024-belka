import lightgbm as lgb
import numpy as np
import polars as pl
import pandas as pd
from modules.utils import listfiles, pad0, save1, fileexists, dircreate, load1
from modules.score import kaggle_score
from modules.mols import features
import os
from datetime import datetime
from distributed import Client, LocalCluster
import dask.array as da

#cluster = LocalCluster(n_workers=8)
#client = Client(cluster)

n_files = 10
run_name = 'files10-onehot'
dircreate('out/gbm')

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
    'max_bin': 100,
    'keep_training_booster': True
}

train_val_blocks = pl.read_parquet('out/train/building_blocks.parquet')
val_data = []
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    model_path = f'out/gbm/gbm-{run_name}-{protein_name}.gbm'

    if not os.path.exists(model_path):

        #dask_model = lgb.DaskLGBMClassifier(client=client)        
        #gbm = lgb.train(
        #    params, 
        #    num_boost_round=20,
        #    callbacks=[lgb.early_stopping(stopping_rounds=15)]
        #)

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
            print('features')
            idt = pl.read_parquet(file)
            iX = features(idt, train_val_blocks)
            iy = idt['binds'].to_numpy()
            lgb_train = lgb.Dataset(iX, iy)
        
            #dask_model.fit(da.from_array(iX), da.from_array(iy))            
            #save1(dask_model, model_path)
            
            print('train')
            if fileexists(model_path):
                gbm = lgb.train(params, lgb_train, init_model = model_path)
            else:
                gbm = lgb.train(params, lgb_train)
            gbm.save_model(model_path)
            
            del file, idt, iX, iy, lgb_train

        del gbm
        

print('validate model')
val_data = []
gbms = {}
for protein_name in ['sEH', 'BRD4', 'HSA']:
        
    model_path = f'out/gbm/gbm-{run_name}-{protein_name}.gbm'
    gbms[protein_name] = lgb.Booster(model_file = model_path)
    
    dofiles = listfiles('out/train/val/base/', protein_name)
    if n_files: dofiles = np.random.choice(dofiles, n_files)

    ct = 0
    for file in dofiles:
        
        ct += 1
        print(f'    {protein_name} {ct} of {len(dofiles)}: {file}')
        idt = pl.read_parquet(file)      
        iX = features(idt, train_val_blocks)
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
test_blocks = pl.read_parquet('out/test/building_blocks.parquet')
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    dofiles = listfiles('out/test/test/base/', protein_name)
    ct = 0
    for file in dofiles:
        
        ct += 1
        print(f'    {protein_name} {ct} of {len(dofiles)}: {file}')
        idt = pl.read_parquet(file)      
        iX = features(idt, test_blocks)

        submission.append(pd.DataFrame({
            'id': pl.read_parquet(file)['id'], 
            'binds': gbms[protein_name].predict(iX)
        }))
        
        del file, idt, iX
    
    del protein_name
    
submission = pd.concat(submission).sort_values('id')
submission.to_parquet(f'out/gbm/submission-{datetime.today().strftime("%Y%m%d")}-lgbm-{run_name}-{pad0(int(expected_score*100))}.parquet', index = False)



