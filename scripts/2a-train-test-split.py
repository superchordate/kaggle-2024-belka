# create train/validation split for testing.

import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from sklearn.model_selection import train_test_split
from modules.utils import dircreate, save1, pad0, write_parquet_from_pyarrow
from modules.mols import get_blocks
import polars as pl
 
dircreate('out/train/train')
dircreate('out/train/val')
dircreate('out/test/test')

# split blocks to train and val.
train_blocks, val_blocks = train_test_split(
    pl.read_parquet('out/train/building_blocks.parquet').to_pandas()[['index', 'ecfp_pca', 'onehot_pca']], 
    test_size = 0.35,
    random_state = 1114 
)
train_blocks.to_parquet('out/train/train/building_blocks.parquet')
val_blocks.to_parquet('out/train/val/building_blocks.parquet')

# save versions of the data containing only the selected blocks for train/val.
mols = pl.read_parquet('out/train/mols.parquet')
for train_val in ['train', 'val']:
    
    print(train_val)
    blocks = train_blocks if train_val == 'train' else val_blocks
    antiblocks = train_blocks if train_val == 'val' else val_blocks
    
    # filter to rows containing the selected blocks.
    imols = mols
    imols = imols.filter(pl.col('buildingblock1_index').is_in(blocks['index']))
    imols = imols.filter(pl.col('buildingblock2_index').is_in(blocks['index']))
    imols = imols.filter(pl.col('buildingblock3_index').is_in(blocks['index']))
    
    # filter to rows not containing val blocks.
    imols = imols.filter(pl.col('buildingblock1_index').is_in(antiblocks['index']).not_())
    imols = imols.filter(pl.col('buildingblock2_index').is_in(antiblocks['index']).not_())
    imols = imols.filter(pl.col('buildingblock3_index').is_in(antiblocks['index']).not_())
    
    # save result. 
    imols.write_parquet(f'out/train/{train_val}/mols.parquet')
    del imols, blocks, antiblocks
        
# test is the same.
print('test')
mols = pl.read_parquet('out/test/mols.parquet')
mols.write_parquet('out/test/test/mols.parquet')
