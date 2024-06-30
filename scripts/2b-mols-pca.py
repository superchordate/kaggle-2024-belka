import polars as pl
from modules.features import features
import numpy as np
from modules.preprocessing import get_pca
from modules.utils import save1, load1, dircreate

# use the test records for PCA.
# this will orient us towards a high score, and reduce memory usage.
blocks = pl.read_parquet('out/blocks-3-pca.parquet')
options = {}

# mols_test = pl.read_parquet('out/test/mols.parquet')
# pipe = get_pca(
#     features(mols_test.sample(10*1000), blocks, options), 
#     info_cutoff = 0.90, verbose = 2
# )
# save1(pipe, 'out/mols-pca.pkl')
pipe = load1('out/mols-pca.pkl')

# apply PCA to test mols.
for train_test in ['train']:

    mols = pl.read_parquet(f'out/{train_test}/mols.parquet')
    nsplits = int(mols.shape[0]/(100*1000))
    mols = mols.with_row_index()
    mols = mols.with_columns((pl.col('index') / mols.shape[0] * nsplits).floor())
    mols = mols.partition_by('index', include_key = False)
    ct = 0
    dircreate(f'out/{train_test}/mols-features')
    outcols = ['molecule_id']
    if train_test == 'train': outcols = outcols + ['binds_sEH', 'binds_BRD4', 'binds_HSA']
    for imols in mols:
        ct += 1
        print(f'split {ct} of {nsplits} {imols.shape[0]:,.0f} rows')
        ifeatures = features(imols, blocks, options)
        ifeatures = pipe.transform(ifeatures)
        ifeatures = np.array(ifeatures * 1000, dtype = np.int8)
        imols.select(outcols).with_columns(pl.Series('features', ifeatures))
        imols.write_parquet(f'out/{train_test}/mols-features/mols-features-{ct}.parquet')
        del imols, ifeatures

