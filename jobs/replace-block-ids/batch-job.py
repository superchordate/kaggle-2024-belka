# molecule data is repeated for each protein.
# we can greatly compress data by merging so each molecule is only stored once.

import polars as pl
import gc, os
import numpy as np

os.system(f'gsutil cp gs://kaggle-417721/blocks-1-smiles.parquet blocks-1-smiles.parquet')

train_test = 'train'

blocks = pl.read_parquet(f'/blocks-1-smiles.parquet')
blocks = blocks.select(['index', 'smiles'])

for protein_name in ['sEH', 'BRD4', 'HSA']:  
    
    filename = f'{train_test}-{protein_name}-wids.parquet'
    
    print(f'replace block ids at: {filename}')
    os.system(f'gsutil cp gs://kaggle-417721/train-{protein_name}.parquet train-{protein_name}.parquet')
    dt = pl.read_parquet(f'train-{protein_name}.parquet')
    dt = dt.with_columns(pl.col('id').cast(pl.UInt32))
    
    # joins in pyarrow will be faster, start there.
    dt = dt.join(blocks, left_on = 'buildingblock1_smiles', right_on = 'smiles', how = 'inner')
    dt = dt.rename({'index': 'buildingblock1_index'}).drop('buildingblock1_smiles')
    
    dt = dt.join(blocks, left_on = 'buildingblock2_smiles', right_on = 'smiles', how = 'inner')
    dt = dt.rename({'index': 'buildingblock2_index'}).drop('buildingblock2_smiles')
    
    dt = dt.join(blocks, left_on = 'buildingblock3_smiles', right_on = 'smiles', how = 'inner')
    dt = dt.rename({'index': 'buildingblock3_index'}).drop('buildingblock3_smiles')

    dt = dt.sort('id')
    
    if (len(dt['id']) != len(np.unique(dt['id']))) or (len(dt['id']) != len(np.unique(dt['id']))):
        raise Exception('rows duplicated')
    
    # update the file.
    dt.write_parquet(filename)    
    os.system(f'gsutil cp {filename} gs://kaggle-417721/{filename}')
    
    del dt, protein_name, filename
    gc.collect()