import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl
import os
from modules.utils import load1

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

# def features(dt, blocks, options):

#     if options['ecfp']:
#         iblocks_ecfp_pca = [np.vstack(blocks['ecfp_pca'][dt[x]]) for x in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']]
#         iblocks_ecfp_pca = np.concatenate(iblocks_ecfp_pca, axis = 1)

#     if options['onehot']:
#         iblocks_onehot_pca = [np.vstack(blocks['onehot_pca'][dt[x]]) for x in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']]
#         iblocks_onehot_pca = np.concatenate(iblocks_onehot_pca, axis = 1)

#     if options['ecfp'] and options['onehot']:
#         return np.concatenate([iblocks_ecfp_pca, iblocks_onehot_pca], axis = 1)
#     elif options['ecfp']:
#         return iblocks_ecfp_pca
#     elif options['onehot']:
#         return iblocks_onehot_pca
