from modules.features import features
from modules.datasets import get_loader
from modules.train import run_val, get_model_optimizer
from modules.utils import gcp, dircreate, kaggle, cloud
from modules.score import kaggle_score, print_results
from datetime import datetime
import pandas as pd
import polars as pl
import torch, os
import numpy as np

# get the network.
options = {
    'lr': 0.001,
    'momentum': 0.9,
    'dropout': 0,
    'network': 'siamese'
}

run_name = 'siamese-BRD4-all-drop50-pca90'

testlen = pl.read_parquet('out/test/mols.parquet').shape[0]
testlen = 50*1000
mols = pl.read_parquet('out/train/val/mols.parquet').sample(testlen)
blocks = pl.read_parquet('out/blocks-3-pca.parquet')

datafolder = 'out/train/val/'
load_path = f'out/net/{run_name}'

net, optimizer = get_model_optimizer(
    options, mols, blocks, 
    load_path = load_path, 
    load_optimizer = False
)
net.eval()

# get embeddings for training data that does not bind.
train_mols_binds = pl.read_parquet('out/train/train/mols.parquet')
blocks = pl.read_parquet('out/blocks-3-pca.parquet')
train_features_binds = features(train_mols_binds.filter(pl.col('binds_BRD4')).sample(5000), blocks, options)
train_features_nobinds = features(train_mols_binds.filter(pl.col('binds_BRD4').not_()).sample(5000), blocks, options)

def prepoutputforcompare(features):
    
    netout = net(
        torch.from_numpy(features[0]).float(), 
        torch.from_numpy(features[1]).float(), 
        torch.from_numpy(features[2]).float()
    )
    
    xd = np.array([float(netout[0][i]) for i in range(len(netout[0]))])
    yd = np.array([float(netout[1][i]) for i in range(len(netout[0]))])
    zd = np.array([float(netout[2][i]) for i in range(len(netout[0]))])
    
    return [xd, yd, zd]
        
def dist(xyzs_context, xyzs_query):
    # [np.mean(bindsx * tox[i] + bindsy * toy[i] + bindsz * toz[i]) for i in range(len(tox))]
    return np.array([
        np.min((
            (xyzs_context[0] - xyzs_query[0][i])**2 + \
            (xyzs_context[1] - xyzs_query[1][i])**2 + \
            (xyzs_context[2] - xyzs_query[2][i])**2
        )**0.5) for i in range(len(xyzs_query[0]))
    ])

print('binding embeddings')
bindsxyz = prepoutputforcompare(train_features_binds)
print('nonbinding embeddings')
nobindsxyz = prepoutputforcompare(train_features_nobinds)
        
for i in range(10):
    print(i)
    
    test_mols = pl.read_parquet('out/train/val/mols.parquet')
    test_mols_binds = test_mols.filter(pl.col('binds_BRD4')).sample(10)
    test_mols_nobinds = test_mols.filter(pl.col('binds_BRD4').not_()).sample(10)
    
    ifeatures = features(test_mols_binds, blocks, options)
    tobinds = prepoutputforcompare(ifeatures)
    print(f'mean dist binds -> binds is: {np.mean(dist(bindsxyz, tobinds)):.4f}')
    print(f'mean dist binds -> nobinds is: {np.mean(dist(nobindsxyz, tobinds)):.4f}')
    
    ifeatures = features(test_mols_nobinds, blocks, options)
    tonobinds= prepoutputforcompare(ifeatures)
    print(f'mean dist nobinds -> binds is: {np.mean(dist(bindsxyz, tonobinds)):.4f}')
    print(f'mean dist nobinds -> nobinds is: {np.mean(dist(nobindsxyz, tonobinds)):.4f}')
    
# get mean distances for val data. 
print('val embeddings')
val_mols = pl.read_parquet('out/train/val/mols.parquet').sample(1000)
val_features = features(val_mols, blocks, options)
valxyz = prepoutputforcompare(val_features)

binds = dist(bindsxyz, valxyz)
nobinds = dist(nobindsxyz, valxyz)

scores = [1 if binds[i] < nobinds[i] else 0 for i in range(len(valxyz[0]))]

scoredt = pd.DataFrame({'bindsscore': binds, 'nobindsscore': nobinds, 'binds': val_mols['binds_BRD4']})
scoredt[scoredt.binds]

scores = [1 if x > 0.5 else 0 for x in dist(nobindsxyz, valxyz)]
print_results(val_mols['binds_BRD4'].cast(pl.Float32), scores)
len(valxyz[0])
