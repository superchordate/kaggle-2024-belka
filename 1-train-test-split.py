# create train/validation split for testing.

import os, pickle
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from modules.utils import dircreate, save1, load1, pad0

fromscratch = False # to reset files in folders.
 
dircreate('out')
for train_test in ['train', 'val']:
    dircreate(f'out/train/{train_test}', fromscratch = fromscratch )
    dircreate(f'out/train/{train_test}/base')
    del train_test
dircreate(f'out/test/test', fromscratch = fromscratch )
dircreate(f'out/test/test/base', fromscratch = fromscratch )

# we need to split by building blocks in order to replicate the contest. 
# so we'll first extract the unique building blocks. 
if False: 
    f = pq.ParquetFile('data/train.parquet')
    batch_size = 10000000
    print(f'{f.metadata.num_rows/1000/1000:.0f} million rows')
    print(f'{f.metadata.num_rows/batch_size:.0f} batches')
    ct = 0
    building_blocks = [[],[],[]]
    for i in f.iter_batches(batch_size = batch_size):
        ct += 1
        building_blocks[0] = np.unique(np.concatenate([building_blocks[0], i['buildingblock1_smiles']]))
        building_blocks[1] = np.unique(np.concatenate([building_blocks[1], i['buildingblock2_smiles']]))
        building_blocks[2] = np.unique(np.concatenate([building_blocks[2], i['buildingblock3_smiles']]))
        print(f'batch {ct} blocks {np.sum([len(x) for x in building_blocks])}')
    save1(building_blocks, 'out/train/building_blocks.pkl')
else:
    building_blocks = load1('out/train/building_blocks.pkl')

# np.sum(np.isin(building_blocks[0], building_blocks[1]))
# np.sum(np.isin(building_blocks[0], building_blocks[2]))
# np.sum(np.isin(building_blocks[1], building_blocks[0]))
# np.sum(np.isin(building_blocks[1], building_blocks[2])) # some overlap here. 
# np.sum(np.isin(building_blocks[2], building_blocks[0]))
# np.sum(np.isin(building_blocks[2], building_blocks[1])) # some overlap here.

# some overlap between 1 and 2. 0 is distinct. 

# let's split the building blocks to train/test.
building_blocks = np.unique(np.concatenate(building_blocks))

train_blocks, test_blocks = train_test_split(building_blocks, test_size = 0.25)
save1(train_blocks, 'out/train/train_blocks.pkl')
save1(test_blocks, 'out/train/test_blocks.pkl')

# now loop over the data and split it.
f = pq.ParquetFile('data/train.parquet')
batch_size = 10*1000*1000
print(f'train/val: {f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/batch_size:.0f} batches')
ct = 0
for i in f.iter_batches(batch_size = batch_size):
    
    ct+= 1
    i = i.to_pandas()
    
    # filter to rows using train blocks and not test blocks.
    i_train = i.merge(pd.DataFrame({'buildingblock1_smiles': train_blocks}), on = 'buildingblock1_smiles', how = 'inner')
    i_train = i_train.merge(pd.DataFrame({'buildingblock2_smiles': train_blocks}), on = 'buildingblock2_smiles', how = 'inner')
    i_train = i_train.merge(pd.DataFrame({'buildingblock3_smiles': train_blocks}), on = 'buildingblock3_smiles', how = 'inner')
    
    i_train  = i_train[~i_train.buildingblock1_smiles.isin(test_blocks)]
    i_train  = i_train[~i_train.buildingblock2_smiles.isin(test_blocks)]
    i_train  = i_train[~i_train.buildingblock3_smiles.isin(test_blocks)]
    
    # similar with test. 
    i_test = i.merge(pd.DataFrame({'buildingblock1_smiles': test_blocks}), on = 'buildingblock1_smiles', how = 'inner')
    i_test = i_test.merge(pd.DataFrame({'buildingblock2_smiles': test_blocks}), on = 'buildingblock2_smiles', how = 'inner')
    i_test = i_test.merge(pd.DataFrame({'buildingblock3_smiles': test_blocks}), on = 'buildingblock3_smiles', how = 'inner')
    
    i_test  = i_test[~i_test.buildingblock1_smiles.isin(train_blocks)]
    i_test  = i_test[~i_test.buildingblock2_smiles.isin(train_blocks)]
    i_test  = i_test[~i_test.buildingblock3_smiles.isin(train_blocks)]
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        i_train[i_train.protein_name == protein_name].to_parquet(f'out/train/train/base/base-{protein_name}-{pad0(ct)}.parquet')
        i_test[i_test.protein_name == protein_name].to_parquet(f'out/train/val/base/base-{protein_name}-{pad0(ct)}.parquet')
        
    del i_test, i_train, i
    
    print(f'finished batch {ct}')

# similar for test but we just save the full test data in batches. 
f = pq.ParquetFile('data/test.parquet')
batch_size = 100000
print(f'test: {f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/batch_size:.0f} batches')
ct = 0
for i in f.iter_batches(batch_size = batch_size):    
    ct+= 1
    i = i.to_pandas()
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        i[i.protein_name == protein_name].to_parquet(f'out/test/test/base/base-{protein_name}-{pad0(ct)}.parquet')
    
    print(f'finished batch {ct}')
