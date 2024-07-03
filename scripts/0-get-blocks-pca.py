import pyarrow.parquet as pq
import polars as pl
import numpy as np
import warnings
from modules.preprocessing import get_pca
from modules.utils import save1, fileremove, dircreate, fileexists
from modules.features import blocks_add_ecfp, blocks_add_onehot, blocks_add_descriptors, blocks_add_graph_embeddings, blocks_add_word_embeddings

fromscratch = False
dircreate('out', fromscratch = fromscratch)
dircreate('out/train')
dircreate('out/test')
dircreate('out/train/blocks')
dircreate('out/test/blocks')

# build initial blocks.
blocks = {}
for train_test in ['test', 'train']:
    
    filename = f'out/{train_test}/blocks/blocks-1-smiles.parquet'
    if fileexists(filename):
        blocks[train_test] = pl.read_parquet(filename)
        del filename
        continue

    f = pq.ParquetFile(f'data/{train_test}.parquet')
    batch_size = 10000000 if train_test == 'train' else f.metadata.num_rows
    print(f'{train_test} {f.metadata.num_rows/1000/1000:.2f} million rows {f.metadata.num_rows/batch_size:,.0f} batches')
    ct = 0
    building_blocks = [[],[],[]]
    for i in f.iter_batches(batch_size = batch_size):
        ct += 1
        print(f'batch {ct}')
        building_blocks[0] = np.unique(np.concatenate([building_blocks[0], i['buildingblock1_smiles']]))
        building_blocks[1] = np.unique(np.concatenate([building_blocks[1], i['buildingblock2_smiles']]))
        building_blocks[2] = np.unique(np.concatenate([building_blocks[2], i['buildingblock3_smiles']]))
        # print(f'batch {ct} {np.sum([len(x) for x in building_blocks]):.0f} unique blocks')
    
    building_blocks = pl.DataFrame({'smiles': np.unique(np.concatenate(building_blocks))})
    building_blocks.write_parquet(filename)
    blocks[train_test] = building_blocks
    
    del train_test, f, i, building_blocks, ct, filename, batch_size

# run preprocessing on full data.
thisfile = f'out/blocks-1-smiles.parquet'
if fileexists(thisfile):
    blocks = pl.read_parquet(thisfile)
else:
    blocks = pl.concat([blocks['train'], blocks['test']]).select(['smiles']).unique()
    
    blocks = blocks.with_row_index()
    blocks = blocks.with_columns(pl.col('index').cast(pl.UInt16))
    
    smiles_enum = pl.Enum(blocks['smiles'])
    blocks = blocks.with_columns(pl.col('smiles').cast(smiles_enum))
    blocks.sort('index').write_parquet(thisfile)
del thisfile

thisfile = f'out/blocks-2-features.parquet'
if fileexists(thisfile):
    blocks = pl.read_parquet(thisfile)
else:
    blocks = blocks_add_ecfp(blocks)
    blocks = blocks_add_onehot(blocks)
    blocks = blocks_add_descriptors(blocks)
    blocks = blocks_add_graph_embeddings(blocks)
    blocks = blocks_add_word_embeddings(blocks)
    blocks.sort('index').write_parquet(thisfile)
del thisfile

# combine features
features = blocks['ecfp'].list.concat(
    blocks['onehot'].list.concat(
        blocks['descrs'].list.concat(
            blocks['graph_embeddings'].list.concat(
                blocks['word_embeddings']
))))
features = np.vstack(features)

# compress with pca.
pcapipe = get_pca(features, info_cutoff = 0.90, from_full = False)
features = pcapipe.transform(features)

# compress to integers.
features = np.array(features * 1000, dtype = np.int8)

# finish and save the block features. 
blocks = blocks.select(['index']).with_columns(pl.Series('features_pca', features))
del pcapipe, features

blocks.sort('index').write_parquet(f'out/blocks-3-pca.parquet')

