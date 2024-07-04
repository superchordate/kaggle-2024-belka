from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch, sys, math
from modules.features import features
from modules.utils import device, cloud
import numpy as np
import polars as pl
    
class Dataset_Mols(Dataset):

    def __init__(self, mols, blocks, targets = None, options = {}):
        
        self.istest = not isinstance(targets, pl.DataFrame)
        self.device = device()
        self.options = options
        print(f'{mols.shape[0]/1000/1000:,.2f} M rows')
        
        if not self.istest:

            targets_torch = {}
            for protein_name in ['sEH', 'BRD4', 'HSA']:
                targets_torch[protein_name] = np.array(targets[f'binds_{protein_name}'].cast(pl.Float32))
                targets_torch[protein_name] = torch.from_numpy(targets_torch[protein_name]).float()
                targets_torch[protein_name] = torch.reshape(targets_torch[protein_name], (-1, 1))
                targets_torch[protein_name] = targets_torch[protein_name].to(self.device)

        mols = mols.select(['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])

        self.features = features(mols, blocks, options)
        print(f'features size: {sys.getsizeof(self.features)/1024/1024/1024:.2f} GB')
        if options['network'] == 'siamese':
            self.features = [
                torch.from_numpy(self.features[0]).float().to(self.device),
                torch.from_numpy(self.features[1]).float().to(self.device),
                torch.from_numpy(self.features[2]).float().to(self.device)
            ]
        else:
            self.features = torch.from_numpy(self.features).float().to(self.device)

        self.mol_ids = mols['molecule_id']
        if not self.istest: self.targets = targets_torch

    def __len__(self):
        return len(self.mol_ids)
    
    def __getitem__(self, idx):

        if not self.istest:
        
            iy = {
                'sEH': self.targets['sEH'][idx],
                'BRD4': self.targets['BRD4'][idx],
                'HSA': self.targets['HSA'][idx]
            }
        
        else:
            
            iy = {'sEH': [], 'BRD4': [], 'HSA': []}
        if self.options['network'] == 'siamese':
            return self.mol_ids[idx], self.features[0][idx], self.features[1][idx], self.features[2][idx], iy
        else:
            return self.mol_ids[idx], self.features[idx], iy
            

def get_loader(indir, mols = None, blocks = None, options = {}, submit = False, checktrain = False):
    
    if mols is None:

        molpath = f'{indir}/mols.parquet'
        print(f'loading {molpath}')
        istest = 'test' in indir
        isval = 'val' in indir
        getcols = ['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']
        if not istest: getcols = getcols + ['binds_sEH', 'binds_BRD4', 'binds_HSA']

        # always read the full file then sample it down.
        mols = pl.read_parquet(molpath, columns = getcols)

        if checktrain or isval:
            mols = mols.sample(100*1000)            
        elif (not submit) and (str(options['n_rows']) != 'all'):
            mols = mols.sample(options['n_rows'])

        mols = mols.with_row_index()
    
        print(f'read {mols.shape[0]/1000/1000:,.2f} M rows')

        # we must use the full blocks (not train/val) to have aligned indexes.
        if blocks is None:
            if cloud():
                blockpath = 'blocks-3-pca.parquet'
            else:
                blockpath = 'out/blocks-3-pca.parquet'
            print(f'blocks: {blockpath}')
            blocks = pl.read_parquet(blockpath, columns = ['index', 'features_pca'])
        
    else: 
        istest = submit
        isval = False
        mols = mols.with_row_index()

    if istest:
        targets = None
        batch_size = 1000
        shuffle = False
    elif isval:
        targets = mols.select(['index', 'binds_sEH', 'binds_BRD4', 'binds_HSA'])
        batch_size = 1000
        shuffle = False
    else:
        targets = mols.select(['index', 'binds_sEH', 'binds_BRD4', 'binds_HSA'])
        batch_size = options['train_batch_size'] if 'train_batch_size' in options else 500
        shuffle = True
    
    return DataLoader(
        Dataset_Mols(
            mols.select(['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']), 
            blocks, targets, options
        ), 
        batch_size = batch_size, shuffle = shuffle, num_workers = 0, drop_last = not submit
    )

# def get_loader_multi(indir):
    
#     print(f'loading {indir}')
    
#     # get base data.
#     dt = []
#     for protein in ['BRD4', 'HSA', 'sEH']:
#         idt = pl.read_parquet(f'{indir}base-sample-{protein}.parquet', columns = ['molecule_smiles', 'binds']).to_pandas()
#         idt['protein_name'] = protein
#         idt = pl.DataFrame(idt)
#         dt.append(idt)
#         del idt, protein
#     dt = pl.concat(dt)
#     dt = dt.select(['protein_name', 'binds', 'molecule_smiles'])    
#     print(f'read {dt.shape[0]} rows')
    
#     # extract multi-task targets.
#     print('extract targets')
#     binds = dt['binds'].to_numpy() == 1
#     BRD4 = (dt['protein_name'] == 'BRD4').to_numpy()
#     sEH = (dt['protein_name'] == 'sEH').to_numpy()
#     binds = np.reshape(binds, (-1, 1))
#     BRD4 = np.reshape(BRD4, (-1, 1))
#     sEH = np.reshape(sEH, (-1, 1))        
#     targets = np.concatenate((binds, BRD4, sEH), axis = 1)
#     del BRD4, sEH, binds
    
#     # build loaders.    
#     loader = LeashDataset(dt.select('molecule_smiles'), targets)
#     batch_size = 100
#     shuffle = True
    
#     return DataLoader(loader, batch_size=batch_size, shuffle=shuffle, num_workers=0), dt['protein_name']

# class LeashDataset(Dataset):    
#     def __init__(self, dt, targets):
#         self.dt = dt
#         self.targets = torch.from_numpy(targets).type(torch.float)
#     def __len__(self):        
#         return self.dt.shape[0]
#     def __getitem__(self, idx):
#         idt = self.dt[idx].map_rows(lambda row: (ecfp(row[0]), ))
#         iecfp = np.array([x for x in idt['column_0']])
#         return torch.from_numpy(iecfp).type(torch.float), self.targets[idx]

# class LeashDataset_Test(Dataset):
#     def __init__(self, dt):
#         self.dt = dt
#     def __len__(self):        
#         return self.dt.shape[0]
#     def __getitem__(self, idx):
#         idt = self.dt[idx].map_rows(lambda row: (ecfp(row[0]), ))
#         iecfp = np.array([x for x in idt['column_0']])
#         return torch.from_numpy(iecfp).type(torch.float)