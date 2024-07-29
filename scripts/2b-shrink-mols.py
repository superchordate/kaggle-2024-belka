# because 99% of molecules do not bind, we can throw most of them out without losing much meaningful information.
# but only in train since we need the others for testing. 

import polars as pl

def reducemols(path):

    mols = pl.read_parquet(f'{path}/mols-all.parquet')
    
    mols_binds = mols.filter(
        pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA')
    )
    
    mols = mols.filter(
        (pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA')).not_()
    )
    #mols_binds.shape[0] # 1.3M binds, I would like a ratio of 10% binds so let's sample 9x this amount.
    
    reduced_mols = pl.concat([mols_binds, mols.sample(mols_binds.shape[0] * 9)])
    
    reduced_mols.write_parquet(f'{path}/mols.parquet')
    

reducemols('out/train/train/')
reducemols('out/train/')