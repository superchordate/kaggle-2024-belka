import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from modules.utils import dircreate, pad0, write_parquet_from_pyarrow, load1
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split

fromscratch = True # to start be clearing out out/train/train and out/train/val.
 
dircreate('out')
for train_test in ['train', 'val']:
    dircreate(f'out/train/{train_test}', fromscratch = fromscratch)
    del train_test
dircreate('out/test/test', fromscratch = fromscratch)

# get smiles for train and val.
train_blocks = pl.read_parquet('out/train/blocks/blocks-1-smiles.parquet')
test_blocks = pl.read_parquet('out/test/blocks/blocks-1-smiles.parquet')

train_train_blocks, train_val_blocks = train_test_split(train_blocks, test_size = 0.1, random_state = 253)
train_train_blocks.sort('index').write_parquet(f'out/train/train/blocks-1-smiles.parquet.parquet')
train_val_blocks.sort('index').write_parquet(f'out/train/val/blocks-1-smiles.parquet.parquet')

# what % of blocks in test are unique to test? 46%.
#1 - test_blocks.filter(pl.col('smiles').is_in(train_blocks['smiles'])).shape[0]/test_blocks.shape[0]

# so in our val, we want it to include 50% blocks not in train. to do this, we'll sample from train.
train_val_distinct_blocks = train_val_blocks
train_val_distinct_blocks.write_parquet('out/train/val/train_val_distinct_blocks.parquet')

# start with train_val being mols that included nonshared blocks.
train_mols = pl.read_parquet('out/train/mols-all.parquet')
train_val_mols = train_mols.filter(
    pl.col('buildingblock1_index').is_in(train_val_blocks['index']) |
    pl.col('buildingblock2_index').is_in(train_val_blocks['index']) |   
    pl.col('buildingblock3_index').is_in(train_val_blocks['index'])
)
train_mols = train_mols.join(train_val_mols, on = 'molecule_id', how = 'anti')

#! nevermind
# now add the same number of rows from training to try for a 50/50 mix.
# train_val_mols = pl.concat([train_val_mols, train_mols.sample(train_val_mols.shape[0])])
# train_mols = train_mols.join(train_val_mols, on = 'molecule_id', how = 'anti')

# we are now at 15% data going to val, with 48% of train val blocks being unique to val.
print(f'{train_val_mols.shape[0] / (train_val_mols.shape[0] + train_mols.shape[0]):.2f} data goes to val.')
train_val_mols_blocks = np.unique(np.concatenate([
    train_val_mols['buildingblock1_index'],
    train_val_mols['buildingblock2_index'],
    train_val_mols['buildingblock3_index'],
]))
print(f'{1 - np.mean(np.isin(train_val_mols_blocks, train_blocks["index"])):.2f} val blocks are unique to val.')

# cool - save the final mols.
train_mols.write_parquet('out/train/train/mols-all.parquet')
train_val_mols.write_parquet('out/train/val/mols.parquet')

