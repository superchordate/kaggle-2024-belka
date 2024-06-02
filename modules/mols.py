from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl
import os
from modules.utils import load1

def ecfp(smile, radius=2, bits=1024):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        return np.full(bits, -1)
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

def get_blocks(train_test):
    
    print('get_blocks')
    pca = load1('out/train/building_blocks-ecfp-pca.pkl')
    filepath = f'out/{train_test}/building_blocks.parquet'
    if not os.path.exists(filepath):
        print(f'did not find {filepath}')

        f = pq.ParquetFile(f'data/{train_test}.parquet')
        batch_size = 10000000 if train_test == 'train' else f.metadata.num_rows
        print(f'{train_test} {f.metadata.num_rows/1000/1000:.2f} million rows {f.metadata.num_rows/batch_size:,.0f} batches')
        ct = 0
        building_blocks = [[],[],[]]
        # molecules = []
        for i in f.iter_batches(batch_size = batch_size):
            ct += 1
            building_blocks[0] = np.unique(np.concatenate([building_blocks[0], i['buildingblock1_smiles']]))
            building_blocks[1] = np.unique(np.concatenate([building_blocks[1], i['buildingblock2_smiles']]))
            building_blocks[2] = np.unique(np.concatenate([building_blocks[2], i['buildingblock3_smiles']]))
            print(f'batch {ct} blocks {np.sum([len(x) for x in building_blocks]):,.0f}')
    
            # molecules = np.unique(np.concatenate([molecules, i['molecule_smiles']]))
            # print(f'molecules {len(molecules):,.0f}')

            # if ct == 1: break
        
        print('adding ecfp')
        building_blocks = pl.DataFrame({'smile': np.unique(np.concatenate(building_blocks))}).with_row_index()
        building_blocks = building_blocks.map_rows(lambda row: (row[0], row[1], ecfp(row[1])))
        building_blocks.columns = ['index', 'smile', 'ecfp']

        print('running PCA')
        ecfps = np.array([x for x in building_blocks['ecfp']])
        building_blocks = building_blocks.with_columns(pl.Series('ecfp_pca', pca.transform(ecfps)))
            
        print(f'writing {filepath}')
        building_blocks.write_parquet(filepath)
    
    # re-read to get a Pyarrow Table.
    blocks = pq.ParquetFile(filepath).read(columns = ['index', 'smile', 'ecfp_pca'])
    # smile comes in as a long string for some reason. fix it to regular string.
    blocks.add_column(1, 'smile', blocks['smile'].cast(pa.string())).remove_column(2)
    return blocks

def add_block_efcps(dt, blocks):

    istest = 'binds' not in dt.columns

    blocks = pl.DataFrame(blocks.select(['index', 'ecfp_pca']).to_pandas())
    dt_othercols = dt.drop(['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
    dt = dt.select(['id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])

    dt = dt.join(blocks, left_on = 'buildingblock1_index', right_on = 'index', how = 'inner').drop('buildingblock1_index').rename({'ecfp_pca': 'ecfp_pca1'})
    dt = dt.join(blocks, left_on = 'buildingblock2_index', right_on = 'index', how = 'inner').drop('buildingblock2_index').rename({'ecfp_pca': 'ecfp_pca2'})
    dt = dt.join(blocks, left_on = 'buildingblock3_index', right_on = 'index', how = 'inner').drop('buildingblock3_index').rename({'ecfp_pca': 'ecfp_pca3'})

    # dt = dt.map_rows(lambda row: (row[0], row[1] + row[2] + row[3]))
    # dt.columns = ['id', 'blocks_ecfp_pca']
    dt = dt.with_columns(pl.col('ecfp_pca1').list.concat(pl.col('ecfp_pca2').list.concat(pl.col('ecfp_pca3'))).cast(pl.Float32).alias('blocks_ecfp_pca')).select(['id', 'blocks_ecfp_pca'])
    dt = dt_othercols.join(dt, on = 'id', how = 'inner')

    if istest:
        return dt.select('id', 'blocks_ecfp_pca')
    else:
        return dt.select('id', 'binds', 'blocks_ecfp_pca')
