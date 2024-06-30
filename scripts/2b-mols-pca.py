import polars as pl

# use all the test records.
mols_test = pl.read_parquet('out/test/mols.parquet')

# sample from train to limit size.
mols_train = pl.read_parquet('out/train/mols.parquet').sample(mols_test.shape[0])

mols = pl.concat([pmols_test, mols_train])



