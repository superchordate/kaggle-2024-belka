import polars as pl
import numpy as np

blocks_ecfp_pca = pl.read_parquet('out/train/building_blocks.parquet', columns = ['ecfp_pca'])['ecfp_pca'].to_numpy()
blocks_ecfp_pca = np.array([list(x) for x in blocks_ecfp_pca])

dt = pl.read_parquet('out/train/train/base/base-sEH-01.parquet')
blocks = dt.select(['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
blocks = blocks[1:5]


blocks_ecfp_pca1 = blocks_ecfp_pca[blocks['buildingblock1_index']]
blocks_ecfp_pca2 = blocks_ecfp_pca[blocks['buildingblock2_index']]
blocks_ecfp_pca3 = blocks_ecfp_pca[blocks['buildingblock3_index']]

blocks_ecfp_pca = np.concatenate([blocks_ecfp_pca1, blocks_ecfp_pca2, blocks_ecfp_pca3], axis = 1)

len(blocks_ecfp_pca[0])

from modules.mols import get_blocks
from chemml.chem import Molecule
 
building_blocks = get_blocks('train', just_testing = True)

mol = Molecule(building_blocks['smiles'], input_type='smiles')