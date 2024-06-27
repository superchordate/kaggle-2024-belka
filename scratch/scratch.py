import polars as pl
import numpy as np

# blocks = pl.read_parquet('out/test/blocks/blocks-4-min.parquet')
# mols = pl.read_parquet('out/test/mols.parquet')
# mols.filter(pl.col('buildingblock1_index').is_in(blocks['index']).not_())['buildingblock1_index']

pl.read_parquet('out/train/blocks/blocks-1-smiles.parquet')
pl.read_parquet('out/test/blocks/blocks-1-smiles.parquet')

pl.read_parquet('out/train/blocks/blocks-3-pca.parquet').shape
pl.read_parquet('out/test/blocks/blocks-3-pca.parquet').shape

pl.read_parquet('out/test/test-BRD4-wids.parquet').dtypes

pl.read_parquet('out/train/val/mols.parquet')
