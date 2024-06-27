import polars as pl
import numpy as np
from modules.utils import save1

def fixmols(indir):
    
    print(f'mols: {indir}')
    
    ifile = f'out/{indir}/mols.parquet'
    
    idt = pl.read_parquet(ifile)
    idt = idt.with_columns(pl.col('molecule_id').cast(pl.Int32))
    idt.write_parquet(ifile)

# mols
fixmols('test')
fixmols('test/test')
fixmols('train')
fixmols('train/train')
fixmols('train/val')
    
def fixbuildingblocks(indir):
    
    print(f'buildingblocks: {indir}')
    
    ifile = f'out/{indir}/blocks/blocks-4-min.parquet'
    
    idt = pl.read_parquet(ifile)
    idt = idt.with_columns(pl.col('index').cast(pl.Int32))
    
    ifeatures = np.array(np.round(np.vstack(idt['features_pca']) * 1000, 0), dtype=np.int8)    
    idt = idt.with_columns(pl.Series('features_pca', ifeatures))
    
    idt.write_parquet(f'out/{indir}/blocks/blocks-4-min.parquet')    
    save1(ifeatures, f'out/{indir}/blocks/blocks-features-np.pkl')

# building blocks.
fixbuildingblocks('test')
fixbuildingblocks('train')
fixbuildingblocks('train/train')
fixbuildingblocks('train/val')

# ids only.
print('idsonly')
for train_test in ['test', 'train']:
    
    print(train_test)
    
    for protein_name in ['sEH', 'BRD4', 'HSA']:
        print(protein_name)
        ifile = f'out/{train_test}/{train_test}-{protein_name}-idsonly.parquet'
        idt = pl.read_parquet(ifile)
        idt = idt.with_columns(pl.col('molecule_id').cast(pl.Int32))
        idt = idt.with_columns(pl.col('id').cast(pl.Int32))
        idt.write_parquet(ifile)
        del ifile, idt, protein_name
    del train_test


