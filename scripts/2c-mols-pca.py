import polars as pl
from modules.features import features
import numpy as np
from modules.pca import get_pca
from modules.utils import save1, load1, dircreate

# use the test records for PCA.
# this will orient us towards a high score, and reduce memory usage.
blocks = pl.read_parquet('out/blocks-3-pca.parquet')

mols_test = pl.read_parquet('out/test/mols.parquet')
pipe = get_pca(
    features(mols_test.sample(50*1000), blocks),
    info_cutoff = 0.90, verbose = 2
)
save1(pipe, 'out/mols-pca.pkl')
pipe = load1('out/mols-pca.pkl')

# apply PCA and save. 
test_features = features(mols_test, blocks)
mols_test = mols_test.with_columns(pl.Series('features_pca', pipe.transform(test_features)))
mols_test.select(['molecule_id', 'features_pca']).write_parquet('out/mols-features-pca.parquet')

# now for train and val. 
for train_val in ['train', 'val']:
    imols = pl.read_parquet(f'out/train/{train_val}/mols.parquet')
    ifeatures = features(imols, blocks)
    imols = imols.with_columns(pl.Series('features_pca', pipe.transform(ifeatures)))
    imols.select(['molecule_id', 'features_pca']).write_parquet(f'out/train/{train_val}/mols-features-pca.parquet')
    del imols, ifeatures, train_val





# apply PCA to test mols.
# for train_test in ['train']:

#     mols = pl.read_parquet(f'out/{train_test}/mols.parquet')
#     nsplits = int(mols.shape[0]/(100*1000))
#     mols = mols.with_row_index()
#     mols = mols.with_columns((pl.col('index') / mols.shape[0] * nsplits).floor())
#     mols = mols.partition_by('index', include_key = False)
#     ct = 0
#     dircreate(f'out/{train_test}/mols-features')
#     outcols = ['molecule_id']
#     if train_test == 'train': outcols = outcols + ['binds_sEH', 'binds_BRD4', 'binds_HSA']
#     for imols in mols:
#         ct += 1
#         print(f'split {ct} of {nsplits} {imols.shape[0]:,.0f} rows')
#         ifeatures = features(imols, blocks)
#         ifeatures = pipe.transform(ifeatures)
#         ifeatures = np.array(ifeatures * 1000, dtype = np.int8)
#         imols.select(outcols).with_columns(pl.Series('features', ifeatures))
#         imols.write_parquet(f'out/{train_test}/mols-features/mols-features-{ct}.parquet')
#         del imols, ifeatures

