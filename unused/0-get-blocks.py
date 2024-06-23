# build blocks dataset.

import pyarrow.parquet as pq
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from modules.utils import save1, fileremove, load1
from modules.mols import ecfp

just_testing = False

if not just_testing:
    fileremove('out/train/building_blocks.parquet')
    fileremove('out/test/building_blocks.parquet')

blocks = {}
for train_test in ['train', 'test']:
# for train_test in ['test']:
    
    print(f'blocks for: {train_test}')
    pca = load1('out/train/building_blocks-ecfp-pca.pkl')
    filepath = f'out/{train_test}/{"test-" if just_testing else ""}building_blocks.parquet'

    # get unique building blocks.
    f = pq.ParquetFile(f'data/{train_test}.parquet')
    batch_size = 10000000 if train_test == 'train' else f.metadata.num_rows
    print(f'{train_test} {f.metadata.num_rows/1000/1000:.2f} million rows {f.metadata.num_rows/batch_size:,.0f} batches')
    ct = 0
    building_blocks = [[],[],[]]
    for i in f.iter_batches(batch_size = batch_size):

        ct += 1
        building_blocks[0] = np.unique(np.concatenate([building_blocks[0], i['buildingblock1_smiles']]))
        building_blocks[1] = np.unique(np.concatenate([building_blocks[1], i['buildingblock2_smiles']]))
        building_blocks[2] = np.unique(np.concatenate([building_blocks[2], i['buildingblock3_smiles']]))
        print(f'batch {ct} blocks {np.sum([len(x) for x in building_blocks]):,.0f}')

        del i
        if just_testing and (ct >= 2): break
        
    print('adding ecfp')
    building_blocks = pl.DataFrame({'smiles': np.unique(np.concatenate(building_blocks))}).with_row_index()
    building_blocks = building_blocks.map_rows(lambda row: (row[0], row[1], ecfp(row[1])))
    building_blocks.columns = ['index', 'smiles', 'ecfp']

    print('running PCA')
    ecfps = np.array([x for x in building_blocks['ecfp']])
    building_blocks = building_blocks.with_columns(pl.Series('ecfp_pca', pca.transform(ecfps)))

    blocks[train_test] = building_blocks
        
    print(f'writing {filepath}')
    building_blocks.write_parquet(filepath)

    del building_blocks
    
train_blocks, val_blocks = train_test_split(blocks['train'].to_pandas(), test_size = 0.35)
save1(train_blocks.sort_values('index'), f'out/train/{"test-" if just_testing else ""}train_blocks.pkl')
save1(val_blocks.sort_values('index'), f'out/train/{"test-" if just_testing else ""}val_blocks.pkl')
