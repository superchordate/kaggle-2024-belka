# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

import os
import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from modules.utils import save1, dircreate
from modules.mols import ecfp
    
dircreate('out')
for train_test in ['train', 'test']:
    dircreate(f'out/{train_test}')
    dircreate(f'out/{train_test}/ecfp')
    dircreate(f'out/{train_test}/base')
    del train_test

# run some training data to fit a pca.
ct = 0
train_test = 'train'

f = pq.ParquetFile(f'data/{train_test}.parquet')
print(f'{f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/1000/1000:.0f} batches')
with Chem.SDWriter(f'out/{train_test}/mols.sdf') as w:
    for i in f.iter_batches(batch_size = 1000000):
        ct += 1
        i = i.to_pandas()
        i = i[i.binds == True]
        print(f'{train_test} batch {ct} rows {i.shape[0]}')  
        smiles = i['molecule_smiles'].values    
        for s in smiles:
            w.write(Chem.MolFromSmiles(s))    
        
    # i.to_parquet(f'out/{train_test}/base/base-{ct}.parquet')
    # ecfp = np.array([generate_ecfp(x) for x in smiles])
    # pca = get_pca(ecfp)
    # save1(f'out/{train_test}/ecfp-pca.pkl')
    # break

# now process test. 
ct = 0
train_test = 'test'

f = pq.ParquetFile(f'data/{train_test}.parquet')
print(f'{f.metadata.num_rows/1000/1000:.0f} million rows')

for i in pq.ParquetFile(f'data/{train_test}.parquet').iter_batches(batch_size = 100000):
    ct += 1
    i = i.to_pandas()
    print(f'batch {ct} rows {i.shape[0]}')
    i.to_parquet(f'out/{train_test}/base/base-{ct}.parquet')    
    ecfp = np.array([generate_ecfp(x) for x in i['molecule_smiles']])
    ecfp = pca(ecfp)        
    pl.DataFrame({'id': i['id'], 'ecfp_pca': ecfp}).write_parquet(f'out/{train_test}/ecfp/ecfp-{ct}.parquet')
    del i, ecfp

