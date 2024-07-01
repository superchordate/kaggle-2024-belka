import polars as pl

t1 = pl.read_parquet('out/submit/submission-20240630-md-allrows-3e-prior.parquet')
t2 = pl.read_parquet('out/submit/submission-20240701-md-allrows-3e.parquet')

t1.filter(pl.col('binds') != t2['binds'])
