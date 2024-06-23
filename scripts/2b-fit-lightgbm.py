import lightgbm as lgb
import numpy as np
import polars as pl
import pandas as pd
from modules.utils import listfiles, pad0, load1
from modules.score import kaggle_score
from modules.mols import add_block_ecfps
import os
from datetime import datetime

n_files = 100
run_name = 'files10-newparams'

params = {
    "objective": "binary",
    "max_depth": 10,
    "min_data_in_leaf": 1000,
    "learning_rate": 0.01,
    "verbose": 1,
    'metric': ['binary'],
    'lambda_l1': 0.3,
    'lambda_l2': 0.3,
    'is_unbalance': True
}


train_blocks = load1('out/train/train_blocks.pkl')
val_blocks = load1('out/train/val_blocks.pkl')
# train_blocks_all = pl.read_parquet('out/train/building_blocks.parquet').to_pandas()
test_blocks = pl.read_parquet('out/test/building_blocks.parquet').to_pandas()

gbms = {}
val_data = []
blocks = pl.read_parquet('out/train/building_blocks.parquet')
for protein_name in ['sEH', 'BRD4', 'HSA']:
    
    model_path = f'out/train/train/gbm-{run_name}-{protein_name}.gbm'

    if not os.path.exists(model_path):

        print(f'Training model for {protein_name}')

        print('read train')
        X_train = []
        y_train = []
        for file in np.random.choice(listfiles('out/train/train/base/', protein_name), n_files):
            print(f'    {file}')
            idt = add_block_ecfps(file, train_blocks).select(['binds', 'ecfp_pca'])
            X_train.append(np.vstack(idt['ecfp_pca'].to_numpy()))
            y_train.append(idt['binds'].to_numpy())
            del file, idt
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        print('read val')
        dt = []
        for file in np.random.choice(listfiles('out/train/val/base/', protein_name), n_files):
            dt.append(add_block_ecfps(file, val_blocks))
            del file
        dt = pl.concat(dt).select(['id', 'binds', 'ecfp_pca'])

        X_val = np.vstack(dt['ecfp_pca'].to_numpy())
        y_val = dt['binds'].to_numpy()
        id_val = dt['id'].to_numpy()
        del dt        
        
        print('train model')
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
        gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)])
        gbm.save_model(model_path)
        gbms[protein_name] = gbm

        del X_train, y_train, gbm, lgb_train, lgb_eval
        
    else:

        gbms[protein_name] = lgb.Booster(model_file = model_path)

        print('read val')
        dt = []
        for file in np.random.choice(listfiles('out/train/val/base/', protein_name), n_files):
            dt.append(add_block_ecfps(file, val_blocks))
            del file
        dt = pl.concat(dt).select(['id', 'binds', 'ecfp_pca'])

        X_val = np.vstack(dt['ecfp_pca'].cast(pl.FLoat32).to_numpy())
        y_val = dt['binds'].to_numpy()
        id_val = dt['id'].to_numpy()
        del dt 

    print('evaluate model')
    scores_val = gbms[protein_name].predict(X_val, num_iteration=gbms[protein_name].best_iteration)

    val_data.append(pd.DataFrame({
        'id': id_val,
        'protein_name': [protein_name]*len(id_val), 
        'binds': y_val,
        'split_group': [1]*len(id_val)
    }))

    del X_val, y_val, scores_val

# check val to get expected score.
print('val')
solution = pd.concat(val_data).sort_values('id')
submission = pd.concat([x[['id', 'binds']] for x in val_data]).sort_values('id')
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

    X_test = np.vstack(dt['ecfp_pca'].to_numpy())
    id_test = dt['id'].to_numpy()
    scores_test = gbms[protein_name].predict(X_test, num_iteration=gbms[protein_name].best_iteration)

    submission.append(pd.DataFrame({
        'id': id_test,
        'binds': scores_test
    }))
    
    del protein_name, X_test, id_test, scores_test, dt
    
submission = pd.concat(submission).sort_values('id')
submission.to_parquet(f'out/submit/submission-{datetime.today().strftime("%Y%m%d")}-lgbm-{run_name}-{pad0(int(expected_score*100))}.parquet', index = False)



