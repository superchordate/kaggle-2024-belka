# see jobs\replace-block-ids\batch-job.py to do train. this script only does test. 
import polars as pl
from modules.utils import fileexists

train_test = 'test'

blocks = pl.read_parquet(f'out/{train_test}/building_blocks.parquet')
blocks = blocks.select(['index', 'smile'])

for protein_name in ['sEH', 'BRD4', 'HSA']:  
    
    filename = f'out/{train_test}/{train_test}-{protein_name}-wids.parquet'
    if not fileexists(filename):   
    
        print(f'replace block ids at: {filename}')
        dt = pl.read_parquet(f'out/{train_test}/{train_test}-{protein_name}.parquet', low_memory = True)
        
        # joins in pyarrow will be faster, start there.
        dt = dt.join(blocks, left_on = 'buildingblock1_smiles', right_on = 'smile', how = 'inner', low_memory = True)
        dt = dt.rename({'index': 'buildingblock1_index'}).drop('buildingblock1_smiles')
        
        dt = dt.join(blocks, left_on = 'buildingblock2_smiles', right_on = 'smile', how = 'inner', low_memory = True)
        dt = dt.rename({'index': 'buildingblock2_index'}).drop('buildingblock2_smiles')
        
        dt = dt.join(blocks, left_on = 'buildingblock3_smiles', right_on = 'smile', how = 'inner', low_memory = True)
        dt = dt.rename({'index': 'buildingblock3_index'}).drop('buildingblock3_smiles')
    
        dt = dt.sort('id')
        
        if (len(dt['id']) != len(np.unique(dt['id']))) or (len(dt['id']) != len(np.unique(dt['id']))):
            raise Exception('rows duplicated')
        
        # update the file.
        dt.write_parquet(filename)
        
        del dt, protein_