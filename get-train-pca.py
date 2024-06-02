import polars as pl
from modules.preprocessing import get_pca
from modules.utils import save1
import numpy as np

blocks = pl.read_parquet('out/train/building_blocks.parquet', columns = ['ecfp'])['ecfp'].to_numpy()
blocks = np.array([list(x) for x in blocks])

pca = get_pca(blocks, from_full = False)

save1(pca, 'out/train/building_blocks-ecfp-pca.pkl')

