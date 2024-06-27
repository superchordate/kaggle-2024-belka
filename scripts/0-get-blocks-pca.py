import pyarrow.parquet as pq
import polars as pl
import numpy as np
import warnings
from modules.preprocessing import get_pca
from modules.utils import save1, fileremove, dircreate, fileexists
from modules.features import blocks_add_ecfp, blocks_add_onehot, blocks_add_descriptors, blocks_add_graph_embeddings, blocks_add_word_embeddings

dircreate('out')
dircreate('out/train')
dircreate('out/train/blocks')
dircreate('out/test/blocks')

# build initial blocks.
blocks = {}
for train_test in ['test', 'train']:
    
    filename = f'out/{train_test}/blocks/blocks-1-smiles.parquet'
    if fileexists(filename):
        blocks[train_test] = pl.read_parquet(filename)
        continue        

    f = pq.ParquetFile(f'data/{train_test}.parquet')
    batch_size = 10000000 if train_test == 'train' else f.metadata.num_rows
    print(f'{train_test} {f.metadata.num_rows/1000/1000:.2f} million rows {f.metadata.num_rows/batch_size:,.0f} batches')
    ct = 0
    building_blocks = [[],[],[]]
    for i in f.iter_batches(batch_size = batch_size):
        ct += 1
        building_blocks[0] = np.unique(np.concatenate([building_blocks[0], i['buildingblock1_smiles']]))
        building_blocks[1] = np.unique(np.concatenate([building_blocks[1], i['buildingblock2_smiles']]))
        building_blocks[2] = np.unique(np.concatenate([building_blocks[2], i['buildingblock3_smiles']]))
        print(f'batch {ct} blocks {np.sum([len(x) for x in building_blocks]):,.0f}')
    
    blocks[train_test] = pl.DataFrame({'smiles': np.unique(np.concatenate(building_blocks))}).with_row_index()
    blocks[train_test].write_parquet(filename)
    del train_test, f, i, building_blocks

# run preprocessing on full data.
blocks['train'] = blocks['train'].with_columns(pl.Series('train_test', ['train']*blocks['train'].shape[0]))
blocks['test'] = blocks['test'].with_columns(pl.Series('train_test', ['test']*blocks['test'].shape[0]))
all_blocks = pl.concat([blocks['train'], blocks['test']]).select(['train_test', 'index', 'smiles'])

all_blocks = blocks_add_ecfp(all_blocks)
all_blocks = blocks_add_onehot(all_blocks)
all_blocks = blocks_add_descriptors(all_blocks)
all_blocks = blocks_add_graph_embeddings(all_blocks)
all_blocks = blocks_add_word_embeddings(all_blocks)

for train_test in ['train', 'test']:
    all_blocks.filter(pl.col('train_test') == train_test) \
        .drop('train_test') \
        .sort('index').write_parquet(f'out/{train_test}/blocks/blocks-2-features.parquet')

# combine features and compress with pca. 
features = all_blocks['ecfp'].list.concat(
    all_blocks['onehot'].list.concat(
        all_blocks['descrs'].list.concat(
            all_blocks['graph_embeddings'].list.concat(
                all_blocks['word_embeddings']
))))

features = np.vstack(features)
pcapipe = get_pca(features, info_cutoff = 0.99, from_full = False)
features = pcapipe.transform(features)
all_blocks = all_blocks.select(['train_test', 'index']).with_columns(pl.Series('features_pca', features))

for train_test in ['train', 'test']:
    all_blocks.filter(pl.col('train_test') == train_test) \
        .drop('train_test') \
        .sort('index').write_parquet(f'out/{train_test}/blocks/blocks-3-pca.parquet')

