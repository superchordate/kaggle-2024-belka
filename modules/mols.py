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

def get_blocks(train_test, return_pyarrow = True):
    
    print('get_blocks')

    # pca may not be available. 
    try:
        pca = load1('out/train/building_blocks-ecfp-pca.pkl')
        foundpca = True
    except:
        foundpca = False

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

            #if ct == 1: break
        
        print('adding ecfp')
        building_blocks = pl.DataFrame({'smile': np.unique(np.concatenate(building_blocks))}).with_row_index()
        building_blocks = building_blocks.map_rows(lambda row: (row[0], row[1], ecfp(row[1])))
        building_blocks.columns = ['index', 'smile', 'ecfp']

        if foundpca: 
            print('running PCA')
            ecfps = np.array([x for x in building_blocks['ecfp']])
            building_blocks = building_blocks.with_columns(pl.Series('ecfp_pca', pca.transform(ecfps)))

        print(f'writing {filepath}')
        building_blocks.write_parquet(filepath)
    
    # re-read to get a Pyarrow Table.
    if return_pyarrow:
        blocks = pq.ParquetFile(filepath).read(columns = ['index', 'smile', 'ecfp_pca'])
        # smile comes in as a long string for some reason. fix it to regular string.
        blocks.add_column(1, 'smile', blocks['smile'].cast(pa.string())).remove_column(2)
        return blocks

# add blocks via melt:
# def add_block_ecfps(file_path, blocks):
    
#     dt = pl.read_parquet(file_path)
#     idcols = ['id'] + (['binds'] if 'binds' in dt.columns else [])

#     # convert to a long data frame. 
#     ldt = dt.select(idcols + ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
#     ldt = ldt.melt(id_vars = idcols, variable_name = 'buildingblock', value_name = 'index')

#     # add ecfp with a join.
#     ldt = ldt.join(pl.DataFrame(blocks[['index', 'ecfp_pca']]), on = 'index', how = 'inner')

#     # convert back to wide.
#     dt = ldt.pivot(index = idcols, columns = 'buildingblock', values = 'ecfp_pca')
#     dt = dt.with_columns(
#         pl.col('buildingblock1_index').list.concat(
#             pl.col('buildingblock2_index').list.concat(
#                 pl.col('buildingblock3_index')
#         )).alias('ecfp_pca')
#     ).select(idcols + ['ecfp_pca'])
    
#     return dt

# old add blocks via join:
# def add_block_efcps(dt, blocks):

#     istest = 'binds' not in dt.columns

#     blocks = pl.DataFrame(blocks.select(['index', 'ecfp_pca']).to_pandas())
#     dt_othercols = dt.drop(['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
#     dt = dt.select(['id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])

#     dt = dt.join(blocks, left_on = 'buildingblock1_index', right_on = 'index', how = 'inner').drop('buildingblock1_index').rename({'ecfp_pca': 'ecfp_pca1'})
#     dt = dt.join(blocks, left_on = 'buildingblock2_index', right_on = 'index', how = 'inner').drop('buildingblock2_index').rename({'ecfp_pca': 'ecfp_pca2'})
#     dt = dt.join(blocks, left_on = 'buildingblock3_index', right_on = 'index', how = 'inner').drop('buildingblock3_index').rename({'ecfp_pca': 'ecfp_pca3'})

#     # dt = dt.map_rows(lambda row: (row[0], row[1] + row[2] + row[3]))
#     # dt.columns = ['id', 'blocks_ecfp_pca']
#     dt = dt.with_columns(pl.col('ecfp_pca1').list.concat(pl.col('ecfp_pca2').list.concat(pl.col('ecfp_pca3'))).cast(pl.Float32).alias('blocks_ecfp_pca')).select(['id', 'blocks_ecfp_pca'])
#     dt = dt_othercols.join(dt, on = 'id', how = 'inner')

#     if istest:
#         return dt.select('id', 'blocks_ecfp_pca')
#     else:
#         return dt.select('id', 'binds', 'blocks_ecfp_pca')

# old add blocks with loop:
# def add_block_ecfps(file_path, blocks):
    
#     dt = pl.read_parquet(file_path)
    
#     blocks_ecfp_pca = []
#     for i in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']:
#         iblocks_ecfp_pca = blocks['ecfp_pca'][dt[i]].values
#         iblocks_ecfp_pca = np.array([list(x) for x in iblocks_ecfp_pca]).astype('float')
#         blocks_ecfp_pca.append(iblocks_ecfp_pca)
#         del iblocks_ecfp_pca, i
        
#     blocks_ecfp_pca = np.concatenate(blocks_ecfp_pca, axis = 1)

#     dt = dt.with_columns(pl.Series('ecfp_pca', blocks_ecfp_pca))
#     dt = dt.drop(['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])

#     # dt.write_parquet(file_path)
    
#     return dt   

def features(dt, blocks, options):

    if options['ecfp']:
        iblocks_ecfp_pca = [np.vstack(blocks['ecfp_pca'][dt[x]]) for x in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']]
        iblocks_ecfp_pca = np.concatenate(iblocks_ecfp_pca, axis = 1)

    if options['onehot']:
        iblocks_onehot_pca = [np.vstack(blocks['onehot_pca'][dt[x]]) for x in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']]
        iblocks_onehot_pca = np.concatenate(iblocks_onehot_pca, axis = 1)

    if options['ecfp'] and options['onehot']:
        return np.concatenate([iblocks_ecfp_pca, iblocks_onehot_pca], axis = 1)
    elif options['ecfp']:
        return iblocks_ecfp_pca
    elif options['onehot']:
        return iblocks_onehot_pca

