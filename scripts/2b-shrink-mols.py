# because 99% of molecules do not bind, we can throw most of them out without losing much meaningful information.
# but only in train since we need the others for testing. 

import polars as pl

train_train_mols = pl.read_parquet('out/train/train/mols.parquet')

train_train_mols_binds = train_train_mols.filter(
    pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA')
)
train_train_mols = train_train_mols.filter(
    (pl.col(f'binds_sEH') | pl.col(f'binds_BRD4') | pl.col(f'binds_HSA')).not_()
)
train_train_mols_binds.shape[0] # 1.3M binds, I would like a ratio of 25% binds so let's sample 3x this amount.

reduced_mols = pl.concat([train_train_mols_binds, train_train_mols.sample(train_train_mols_binds.shape[0] * 3)])

reduced_mols.write_parquet('out/train/train/mols-reduced.parquet')
