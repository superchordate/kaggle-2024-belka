import polars as pl
import gc

# for train, things are aligned so no join is necessary.
train_test = 'train'
    
mols = pl.read_parquet(f'out/{train_test}/{train_test}-sEH-wids.parquet')

mols = mols.with_columns(pl.col('buildingblock1_index').cast(pl.Int32))
mols = mols.with_columns(pl.col('buildingblock2_index').cast(pl.Int32))
mols = mols.with_columns(pl.col('buildingblock3_index').cast(pl.Int32))
mols = mols.drop('id')
    
mols = mols.rename({'binds': 'binds_sEH'})        
mols = mols.with_columns(pl.col('binds_sEH').cast(pl.Binary))
    
for protein_name in ['BRD4', 'HSA']:
    
    print(protein_name)
       
    idt = pl.read_parquet(f'out/{train_test}/{train_test}-{protein_name}-wids.parquet', columns = ['molecule_smiles', 'binds']) 
    idt = idt.with_columns(pl.col('binds').cast(pl.Binary))

    # validate mols are in the exact same order.
    molscheck = mols.select(['molecule_smiles']).with_columns(pl.Series('molecule_smiles_check', idt['molecule_smiles']))
    if molscheck.filter(pl.col('molecule_smiles') != pl.col('molecule_smiles_check')).shape[0] > 0:
        raise Exception('Mols are not aligned.')
    del molscheck
    gc.collect()
    
    idt = idt.drop('molecule_smiles') 
    
    mols = mols.with_columns(pl.Series(f'binds_{protein_name}', idt['binds']))    
    
    del idt, protein_name
    gc.collect()
    
    mols = mols.with_row_index()
    mols = mols.rename({'index': 'molecule_id'})
    
mols = mols.select([
    'molecule_id',
    'molecule_smiles',
    'buildingblock1_index',
    'buildingblock2_index',
    'buildingblock3_index',
    'binds_sEH',
    'binds_BRD4',
    'binds_HSA'
])

mols.write_parquet(f'out/{train_test}/mols.parquet')
    
# now we can replace protein files with simple files with id and row_index as molecule id.
for protein_name in ['sEH', 'BRD4', 'HSA']:       
    idt = pl.read_parquet(f'out/{train_test}/{train_test}-{protein_name}-wids.parquet', columns = ['id', 'binds'])
    idt = idt.with_row_index()
    idt = idt.with_columns(pl.col('binds').cast(pl.Binary))
    idt = idt.rename({'index': 'molecule_id'})
    idt.write_parquet(f'out/{train_test}/{train_test}-{protein_name}-idsonly.parquet')
    
# for test, no such luck. instead, we'll read them all and take unique to cover all of them.
train_test = 'test'  
mols = [] 
for protein_name in ['sEH', 'BRD4', 'HSA']:
    idt = pl.read_parquet(f'out/{train_test}/{train_test}-{protein_name}-wids.parquet')
    
    idt = idt.with_columns(pl.col('buildingblock1_index').cast(pl.Int32))
    idt = idt.with_columns(pl.col('buildingblock2_index').cast(pl.Int32))
    idt = idt.with_columns(pl.col('buildingblock3_index').cast(pl.Int32))
    idt = idt.drop(['id', 'protein_name'])
    
    mols.append(idt)
    del idt, protein_name
    
mols = pl.concat(mols).unique()
mols = mols.with_row_index().rename({'index': 'molecule_id'})
    
mols = mols.select([
    'molecule_id',
    'molecule_smiles',
    'buildingblock1_index',
    'buildingblock2_index',
    'buildingblock3_index'
])

mols.write_parquet(f'out/{train_test}/mols.parquet')

# now replace with ids.
for protein_name in ['sEH', 'BRD4', 'HSA']:
    idt = pl.read_parquet(
        f'out/{train_test}/{train_test}-{protein_name}-wids.parquet', 
        columns = ['id', 'molecule_smiles']
    )
    idt = idt.join(mols.select(['molecule_smiles', 'molecule_id']), on = 'molecule_smiles', how = 'left')
    idt = idt.drop('molecule_smiles')
    idt.write_parquet(f'out/{train_test}/{train_test}-{protein_name}-idsonly.parquet')

