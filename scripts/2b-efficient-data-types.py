import polars as pl
import numpy as np
from modules.utils import save1

def fixmols(indir):
    
    print(f'mols: {indir}')
    
    ifile = f'out/{indir}/mols.parquet'
    
    idt = pl.read_parquet(ifile)
    idt = idt.with_columns(pl.col('molecule_id').cast(pl.Int32))
    idt.write_parquet(ifile)
    
def fixbuildingblocks(indir):
    
    print(f'buildingblocks: {indir}')
    
    ifile = f'out/{indir}/building_blocks.parquet'
    
    idt = pl.read_parquet(ifile)
    idt = idt.with_columns(pl.col('index').cast(pl.Int32))
    idt = idt.select(['index', 'ecfp_pca', 'onehot_pca'])
    
    iecfp_pca = np.array(np.round(np.vstack(idt['ecfp_pca']) * 100, 0), dtype=np.int8)
    ionehot_pca = np.array(np.round(np.vstack(idt['onehot_pca']) * 100, 0), dtype=np.int8)
    
    idt = idt.with_columns(pl.Series('ecfp_pca', iecfp_pca))    
    idt = idt.with_columns(pl.Series('onehot_pca', ionehot_pca))
    idt.write_parquet(f'out/{indir}/building_blocks-min.parquet')
    
    ifeatures = np.concatenate([iecfp_pca, ionehot_pca], axis = 1)
    idt = idt.with_columns(pl.Series('features', ifeatures))
    idt = idt.select('index', 'features')
    
    idt.write_parquet(f'out/{indir}/building_blocks-features.parquet')  
    save1(ifeatures, f'out/{indir}/building_blocks-features-np.pkl')

# mols
# fixmols('test')
# fixmols('test/test')
# fixmols('train')
# fixmols('train/train')
# fixmols('train/val')

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


