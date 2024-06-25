# molecule data is repeated for each protein.
# we can greatly compress data by merging so each molecule is only stored once.

import polars as pl
import pyarrow.parquet as pq
import gc
import numpy as np
from modules.utils import fileexists

# split the main table into 3 sub-tables, one for each protein.
for train_test in ['test', 'train']:
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        
        filename = f'out/{train_test}/{train_test}-{protein_name}.parquet'
        
        if not fileexists(filename):        
            print(f'create: out/{train_test}/{train_test}-{protein_name}.parquet')            
            columns = ['id', 'buildingblock1_smiles', 'buildingblock2_smiles', 'buildingblock3_smiles', 'molecule_smiles']
            if train_test == 'train': columns.append('binds')
            pq.write_table(pq.read_table(
                source = f'data/{train_test}.parquet',
                columns = columns,
                filters=[('protein_name', '=', protein_name)]
            ), filename)

gc.collect()

