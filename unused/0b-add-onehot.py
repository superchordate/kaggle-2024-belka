# from smiles_encoder import SmilesEncoder
from modules.utils import save1
from modules.preprocessing import get_pca
import polars as pl
import numpy as np
import pandas as pd

trainblockspath = 'out/train/building_blocks.parquet'
building_blocks = pl.read_parquet(trainblockspath)

# encoder = SmilesEncoder(building_blocks['smile'])
# onehots = encoder.encode_many(building_blocks['smile'])
# decoded_smiles = encoder.decode_many(onehots)
# onehots = [unlist(x) for x in onehots]
# pca = get_pca(onehots, from_full = False)

# building_blocks = building_blocks.with_columns(pl.Series('onehot', onehots))
# building_blocks.write_parquet(trainblockspath)

# len(building_blocks['onehot'][0])
# len(building_blocks['onehot'][0][0])

from pysmilesutils.tokenize import SMILESTokenizer

tokenizer = SMILESTokenizer(encoding_type = "one hot")
tokenizer.create_vocabulary_from_smiles(building_blocks['smile'])

smiles_encoded = tokenizer(building_blocks['smile'], enclose = False)
smiles_encoded = [pd.DataFrame(x).apply(sum).values for x in smiles_encoded]
smiles_encoded = np.array([list(x) for x in smiles_encoded])
pca = get_pca(smiles_encoded, from_full = False)

building_blocks = building_blocks.with_columns(pl.Series('onehot_pca', pca.transform(smiles_encoded)))
building_blocks.write_parquet(trainblockspath)
save1(tokenizer, 'out/train/building_blocks-onehot-tokenizer.pkl')
save1(pca, 'out/train/building_blocks-onehot-pca.pkl')

# apply this to test. 
testblockspath = 'out/test/building_blocks.parquet'
building_blocks = pl.read_parquet(testblockspath)
smiles_encoded = tokenizer(building_blocks['smile'], enclose = False)
smiles_encoded = [pd.DataFrame(x).apply(sum).values for x in smiles_encoded]
smiles_encoded = np.array([list(x) for x in smiles_encoded])
building_blocks = building_blocks.with_columns(pl.Series('onehot_pca', pca.transform(smiles_encoded)))
building_blocks.write_parquet(testblockspath)
