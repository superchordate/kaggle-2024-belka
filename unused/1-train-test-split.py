# build blocks dataset.

import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
from modules.utils import dircreate, pad0, write_parquet_from_pyarrow, load1
import polars as pl
import pandas as pd

fromscratch = True # to reset files in folders.
 
dircreate('out')
for train_test in ['train', 'val']:
    dircreate(f'out/train/{train_test}', fromscratch = fromscratch)
    dircreate(f'out/train/{train_test}/base')
    del train_test
dircreate('out/test/test', fromscratch = fromscratch)
dircreate('out/test/test/base', fromscratch = fromscratch)

train_blocks = load1('out/train/train_blocks.pkl')
val_blocks = load1('out/train/val_blocks.pkl')
train_blocks_all = pl.read_parquet('out/train/building_blocks.parquet').to_pandas()
test_blocks = pl.read_parquet('out/test/building_blocks.parquet').to_pandas()

def get_block_indexes(x, blocks, anti_blocks = None):
    
    x = pa.Table.from_batches([x]).remove_column(4) # we won't be using molecule_smiles.
    blocks = pa.Table.from_pandas(blocks)
    
    # add binds dummy col if it is missing.
    if 'binds' not in x.schema.names: x = x.add_column(5, 'binds', [np.array([0]*x.num_rows)])
    blocks = blocks.select(['index', 'smiles'])
    
    # joins in pyarrow will be faster, start there.
    x = x.join(blocks, keys = ['buildingblock1_smiles'], right_keys = ['smiles'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'buildingblock2_smiles', 'buildingblock3_smiles', 'protein_name', 'binds', 'buildingblock1_smiles'])
    
    x = x.join(blocks, keys = ['buildingblock2_smiles'], right_keys = ['smiles'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'buildingblock3_smiles', 'protein_name', 'binds', 'buildingblock1_smiles', 'buildingblock2_smiles'])
    
    x = x.join(blocks, keys = ['buildingblock3_smiles'], right_keys = ['smiles'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'protein_name', 'binds', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
    
    if isinstance(anti_blocks, pd.DataFrame):
        anti_blocks = pa.Table.from_pandas(anti_blocks)
        anti_blocks = anti_blocks.select(['index'])
        x = x.join(anti_blocks, keys = ['buildingblock1_index'], right_keys = ['index'], join_type = 'left anti')
        x = x.join(anti_blocks, keys = ['buildingblock2_index'], right_keys = ['index'], join_type = 'left anti')
        x = x.join(anti_blocks, keys = ['buildingblock3_index'], right_keys = ['index'], join_type = 'left anti')
        
    x = x.sort_by('id')    
    
    if (len(x['id']) != len(np.unique(x['id']))) or (len(x['id']) != len(np.unique(x['id']))):
        raise Exception('rows duplicated')
    
    return x

if not os.path.exists('out/train/train/base/base-BRD4-01.parquet'):

    # now loop over the data to split it and swap block strings for ids.
    f = pq.ParquetFile('data/train.parquet')
    batch_size = 1000*1000
    print(f'train/val: {f.metadata.num_rows/1000/1000:.0f} million rows')
    print(f'{f.metadata.num_rows/batch_size:.0f} batches of {batch_size/1000/1000:.0f} million rows each (includes all proteins)')
    ct = 0

    for i in f.iter_batches(batch_size = batch_size):
        
        ct+= 1
        
        # convert smiles into index references. 
        i_train = get_block_indexes(i, train_blocks, val_blocks)
        i_val = get_block_indexes(i, val_blocks, train_blocks)
        
        # save and add ecfp_pca for ach. 
        for protein_name in ['sEH', 'BRD4', 'HSA']:        

            if i_train.shape[0] > 0:
                ifile_train = f'out/train/train/base/base-{protein_name}-{pad0(ct)}.parquet'
                write_parquet_from_pyarrow(i_train.filter(pc.field('protein_name') == protein_name), ifile_train)
                # add_block_ecfps(ifile_train, train_blocks)

            if i_val.shape[0] > 0:
                ifile_val = f'out/train/val/base/base-{protein_name}-{pad0(ct)}.parquet'
                write_parquet_from_pyarrow(i_val.filter(pc.field('protein_name') == protein_name), ifile_val)
                # add_block_ecfps(ifile_val, val_blocks)
        
        print(f'batch {ct}: {i_train.shape[0]/1000/1000:,.2f} M train rows {i_val.shape[0]:,.0f} val rows')
            
        del i_val, i_train, i

# similar for test but we just save the full test data in batches.
f = pq.ParquetFile('data/test.parquet')
batch_size = 10*1000
print(f'test: {f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/batch_size:.0f} batches')
ct = 0
for i in f.iter_batches(batch_size = batch_size):
    
    ct+= 1
    init_ids = np.array(i['id'])
    i_test = get_block_indexes(i, test_blocks)
    
    # if any(init_ids != i.select(['id']).to_pandas().id.values):
    if np.any(init_ids != np.array(i['id'])):
        raise Exception('Lost rows.')
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        ifile_test = f'out/test/test/base/base-{protein_name}-{pad0(ct)}.parquet'
        write_parquet_from_pyarrow(i_test.filter(pc.field('protein_name') == protein_name), ifile_test)
        # add_block_ecfps(ifile_val, test_blocks)
    
    print(f'batch {ct}: {i_test.shape[0]/1000/1000:,.2f} M test rows')
    
    del i_test, i

