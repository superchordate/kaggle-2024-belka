import polars as pl
from modules.preprocessing import get_pca
from modules.utils import save1, fileremove, dircreate
from modules.mols import get_blocks
import numpy as np

trainblockspath = 'out/train/building_blocks.parquet'

# remove prior files. 
fileremove(trainblockspath)
dircreate('out')
dircreate('out/train')
dircreate('out/test')

# get initial blocks, without PCA.
get_blocks('train', return_pyarrow = False)

# read the blocks and fit PCA. 
print('Fit PCA')
blocks = pl.read_parquet('out/train/building_blocks.parquet', columns = ['ecfp'])['ecfp'].to_numpy()
blocks = np.array([list(x) for x in blocks])
pca = get_pca(blocks, from_full = False)
save1(pca, 'out/train/building_blocks-ecfp-pca.pkl')

# run the fitted PCA on the training blocks. 
print('Running PCA')
building_blocks = pl.read_parquet(trainblockspath)
ecfps = np.array([x for x in building_blocks['ecfp']])
building_blocks = building_blocks.with_columns(pl.Series('ecfp_pca', pca.transform(ecfps)))
building_blocks.write_parquet(trainblockspath)

# we now have our train PCA and blocks. run the test get_blocks.
# this will build the test blocks and add pca. 
get_blocks('test', return_pyarrow = False)

