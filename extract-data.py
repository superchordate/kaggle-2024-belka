# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

import os
import polars as pl
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.decomposition import PCA

def dircreate(x): 
    if not os.path.exists(x):
        os.makedirs(x)
        
def save1(obj, path):
    output = open(path, 'wb')
    pickle.dump(obj, output)
    output.close()
    
dircreate('out')
for train_test in ['train', 'test']:
    dircreate(f'out/{train_test}')
    dircreate(f'out/{train_test}/ecfp')
    dircreate(f'out/{train_test}/base')
    del train_test

def get_pca(X, info_cutoff = 0.95):
    
    print(f'running PCA on {len(X[0])} columns {len(X)} rows')

    # start at number cols minus one and drop columns until you get to 95% of the information.
    n_components = len(X[0])
    if n_components > 1 and n_components < len(X):

        while True:
            pca = PCA(n_components = n_components)
            pca = pca.fit(X)
            if np.sum(pca.explained_variance_ratio_) < (1 - info_cutoff):
                break
            else:
                n_components = n_components - 1
                
        # fit the final pca.
        n_components += 1        
        pca = PCA(n_components = n_components).fit(X)
        
        print({
            'starting cols': len(X[0]), 
            'ending cols': n_components, 
            'explained': round(np.sum(pca.explained_variance_ratio_), 2)
        })
        
        return pca

# process a parquet file by row splits. 

def generate_ecfp(smile, radius=2, bits=1024):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        return np.full(bits, -1)
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

# run some training data to fit a pca.
ct = 0
train_test = 'train'

f = pq.ParquetFile(f'data/{train_test}.parquet')
print(f'{f.metadata.num_rows/1000/1000:.0f} million rows')
print(f'{f.metadata.num_rows/1000/1000:.0f} batches')
with Chem.SDWriter(f'out/{train_test}/mols.sdf') as w:
    for i in f.iter_batches(batch_size = 1000000):
        ct += 1
        i = i.to_pandas()
        i = i[i.binds == True]
        print(f'{train_test} batch {ct} rows {i.shape[0]}')  
        smiles = i['molecule_smiles'].values    
        for s in smiles:
            w.write(Chem.MolFromSmiles(s))    
        
    # i.to_parquet(f'out/{train_test}/base/base-{ct}.parquet')
    # ecfp = np.array([generate_ecfp(x) for x in smiles])
    # pca = get_pca(ecfp)
    # save1(f'out/{train_test}/ecfp-pca.pkl')
    # break

# now process test. 
ct = 0
train_test = 'test'

f = pq.ParquetFile(f'data/{train_test}.parquet')
print(f'{f.metadata.num_rows/1000/1000:.0f} million rows')

for i in pq.ParquetFile(f'data/{train_test}.parquet').iter_batches(batch_size = 100000):
    ct += 1
    i = i.to_pandas()
    print(f'batch {ct} rows {i.shape[0]}')
    i.to_parquet(f'out/{train_test}/base/base-{ct}.parquet')    
    ecfp = np.array([generate_ecfp(x) for x in i['molecule_smiles']])
    ecfp = pca(ecfp)        
    pl.DataFrame({'id': i['id'], 'ecfp_pca': ecfp}).write_parquet(f'out/{train_test}/ecfp/ecfp-{ct}.parquet')
    del i, ecfp

