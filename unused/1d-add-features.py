import polars as pl
import numpy as np
from modules.features import features
from modules.utils import dircreate, pad0

mols = pl.read_parquet('out/train/mols.parquet')
mols = mols.unique()
mols.shape

num_splits = 100

mols = mols.with_columns(pl.Series('group', np.random.choice(range(num_splits), mols.shape[0])))
mols = mols.partition_by('group', include_key = False)
print(f'split mols to {num_splits} random splits for processing.')

blocks = pl.read_parquet('out/blocks-3-pca.parquet')
ct = 0
dircreate('out/train/mols-features')
for imols in mols:
    ct+=1
    ifeatures = features(imols, blocks)
    imols = imols.with_columns(pl.Series('features', ifeatures))
    imols = imols.select(['molecule_id', 'binds_sEH', 'binds_BRD4', 'binds_HSA', 'features'])
    imols.write_parquet(f'out/train/mols-features/mols-features-{pad0(ct)}')
    imols.shape

