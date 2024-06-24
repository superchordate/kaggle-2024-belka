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

fromscratch = False # to reset files in folders.
 
dircreate('out')
for train_test in ['train', 'val']:
    dircreate(f'out/train/{train_test}', fromscratch = fromscratch )
    dircreate(f'out/train/{train_test}/base')
    del train_test
dircreate('out/test/test', fromscratch = fromscratch)
dircreate('out/test/test/base', fromscratch = fromscratch)

def get_block_indexes(x, blocks, anti_blocks = None):
    
    x = pa.Table.from_batches([x]).remove_column(4) # we won't be using molecule_smiles.
    
    # add binds dummy col if it is missing.
    if 'binds' not in x.schema.names: x = x.add_column(5, 'binds', [np.array([0]*x.num_rows)])
    blocks = blocks.select(['index', 'smile'])
    
    # joins in pyarrow will be faster, start there.
    x = x.join(blocks, keys = ['buildingblock1_smiles'], right_keys = ['smile'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'buildingblock2_smiles', 'buildingblock3_smiles', 'protein_name', 'binds', 'buildingblock1_smiles'])
    
    x = x.join(blocks, keys = ['buildingblock2_smiles'], right_keys = ['smile'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'buildingblock3_smiles', 'protein_name', 'binds', 'buildingblock1_smiles', 'buildingblock2_smiles'])
    
    x = x.join(blocks, keys = ['buildingblock3_smiles'], right_keys = ['smile'], join_type = 'inner')
    x = x.remove_column(1)
    x = x.rename_columns(['id', 'protein_name', 'binds', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
    
    if anti_blocks: 
        anti_blocks = anti_blocks.select(['index'])
        x = x.join(anti_blocks, keys = ['buildingblock1_index'], right_keys = ['index'], join_type = 'left anti')
        x = x.join(anti_blocks, keys = ['buildingblock2_index'], right_keys = ['index'], join_type = 'left anti')
        x = x.join(anti_blocks, keys = ['buildingblock3_index'], right_keys = ['index'], join_type = 'left anti')
        
    x = x.sort_by('id')
    
    if (len(x['id']) != len(np.unique(x['id']))) or (len(x['id']) != len(np.unique(x['id']))):
        raise Exception('rows duplicated')
    
    return x

if not os.path.exists('out/train/train/base/base-BRD4-01.parquet'):

    # we need to split by building blocks in order to replicate the contest. 
    # so we'll first extract the unique building blocks. 
    building_blocks = get_blocks('train')
    train_blocks, test_blocks = train_test_split(building_blocks.to_pandas(), test_size = 0.35)
    save1(train_blocks, 'out/train/train_blocks.pkl')
    save1(test_blocks, 'out/train/test_blocks.pkl')

    # now loop over the data to split it and swap block strings for ids.
    f = pq.ParquetFile('data/train.parquet')
    batch_size = 10*1000*1000
    print(f'train/val: {f.metadata.num_rows/1000/1000:.0f} million rows')
    print(f'{f.metadata.num_rows/batch_size:.0f} batches of {batch_size/1000/1000:.0f} million rows each')
    train_blocks = pa.Table.from_pandas(train_blocks, preserve_index = False)
    test_blocks = pa.Table.from_pandas(test_blocks, preserve_index = False)
    ct = 0

    for i in f.iter_batches(batch_size = batch_size):
        
        ct+= 1
        
        i_train = get_block_indexes(i, train_blocks, test_blocks)
        i_test = get_block_indexes(i, test_blocks, train_blocks)
        
        for protein_name in ['sEH', 'BRD4', 'HSA']:

            write_parquet_from_pyarrow(i_train.filter(pc.field('protein_name') == protein_name), f'out/train/train/base/base-{protein_name}-{pad0(ct)}.parquet')
            write_parquet_from_pyarrow(i_test.filter(pc.field('protein_name') == protein_name), f'out/train/val/base/base-{protein_name}-{pad0(ct)}.parquet')

            # now read to polars and add ecfps.
            # print(f'adding ecfp for {protein_name}')
            # add_block_efcps(
            #     pl.read_parquet(f'out/train/train/base/base-{protein_name}-{pad0(ct)}.parquet'), 
            #     train_blocks
            # ).write_parquet(f'out/train/train/base/base-{protein_name}-{pad0(ct)}.parquet')

            # add_block_efcps(
            #     pl.read_parquet(f'out/train/val/base/base-{protein_name}-{pad0(ct)}.parquet'), 
            #     test_blocks
            # ).write_parquet(f'out/train/val/base/base-{protein_name}-{pad0(ct)}.parquet')
        
        print(f'batch {ct}: {i_train.shape[0]/1000/1000:,.2f} M train rows {i_test.shape[0]:,.0f} val rows')
            
        del i_test, i_train, i

# similar for test but we just save the full test data in batches.
building_blocks = get_blocks('test')
building_blocks = building_blocks.add_column(1, 'smile', building_blocks['smile'].cast(pa.string())).remove_column(2)
f = pq.ParquetFile('data/test.parquet')
batch_size = 100000
print(f'test: {f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/batch_size:.0f} batches')
ct = 0
for i in f.iter_batches(batch_size = batch_size):
    
    ct+= 1
    init_ids = np.array(i['id'])
    i = get_block_indexes(i, building_blocks)
    
    # if any(init_ids != i.select(['id']).to_pandas().id.values):
    if np.any(init_ids != np.array(i['id'])):
        raise Exception('Lost rows.')
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        write_parquet_from_pyarrow(i.filter(pc.field('protein_name') == protein_name), f'out/test/test/base/base-{protein_name}-{pad0(ct)}.parquet')
        # print('adding ecfp')
        # add_block_efcps(
        #     pl.read_parquet(f'out/test/test/base/base-{protein_name}-{pad0(ct)}.parquet'), 
        #     building_blocks
        # ).write_parquet(f'out/test/test/base/base-{protein_name}-{pad0(ct)}.parquet')
    
    print(f'batch {ct}')
