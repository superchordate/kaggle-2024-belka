from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch, sys
from modules.mols import features
import numpy as np
import polars as pl
    
class Dataset_Mols(Dataset):

    def __init__(self, mols, blocks, targets = None, device = 'cpu', options = {}):
        
        self.istest = not isinstance(targets, pl.DataFrame)
        
        if not self.istest :
            targets = targets.with_columns(pl.col('binds_sEH').cast(pl.Float32))
            targets = targets.with_columns(pl.col('binds_BRD4').cast(pl.Float32))
            targets = targets.with_columns(pl.col('binds_HSA').cast(pl.Float32))

        mols = mols.select(['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])

        print('getting features')
        self.device = device
        self.features = torch.from_numpy(features(mols, blocks, options)).type(torch.float).to(self.device)
        print(sys.getsizeof(object))

        self.mol_ids = mols['molecule_id']
        self.options = options
        self.targets = targets
        self.len = mols.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):

        # idt = self.mols[idx]
        # iX = features(idt, self.blocks, self.options)

        if not self.istest:
            
            itargets = self.targets[idx]
        
            iy = {
                'sEH': torch.from_numpy(itargets['binds_sEH'].to_numpy()).to(self.device),
                'BRD4': torch.from_numpy(itargets['binds_BRD4'].to_numpy()).to(self.device),
                'HSA': torch.from_numpy(itargets['binds_HSA'].to_numpy()).to(self.device)
            }
        
        else:
            
            iy = {'sEH': [], 'BRD4': [], 'HSA': []}
    
        return self.mol_ids[idx], self.features[idx], iy
            

def get_loader(indir, device = 'cpu',  options = {}, submit = False, checktrain = False):
    
    molpath = f'{indir}/mols.parquet'
    print(f'loading {molpath}')
    istest = 'test' in indir
    isval = 'val' in indir
    getcols = ['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']
    if not istest: getcols = getcols + ['binds_sEH', 'binds_BRD4', 'binds_HSA']

    # always read the full file then sample it down. 
    mols = pl.read_parquet(molpath, columns = getcols)

    if (not submit) and (str(options['n_rows']) != 'all'):
        mols = mols.sample(options['n_rows'])
    elif checktrain:
        mols = mols.sample(100*1000)
    print(f'read {mols.shape[0]/1000/1000:,.2f} M rows')

    # we must use the full blocks (not train/val) to have aligned indexes.
    blockpath = 'out/' + ('test' if istest else 'train') + '/building_blocks-features.parquet'
    print(f'blocks: {blockpath}')
    blocks = pl.read_parquet(blockpath, columns = ['index', 'features'])

    if istest:
        targets = None
        batch_size = 5*1000
        shuffle = False
    elif isval:
        targets = mols.select(['binds_sEH', 'binds_BRD4', 'binds_HSA'])
        batch_size = 5*1000
        shuffle = False
    else:
        targets = mols.select(['binds_sEH', 'binds_BRD4', 'binds_HSA'])
        batch_size = options['train_batch_size']
        shuffle = True
    
    return DataLoader(
        Dataset_Mols(
            mols.select(['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']), 
            blocks, targets, device, options
        ), 
        batch_size = batch_size, shuffle = shuffle, num_workers = 0
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