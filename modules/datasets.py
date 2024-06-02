from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from modules.mols import ecfp
from modules.utils import listfiles
import numpy as np
import polars as pl
    
class Dataset_Blocks(Dataset):

    def __init__(self, dt, blocks, targets = None):
        self.ids = dt['id'].to_numpy()
        self.blocks = dt.select(['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'])
        self.blocks_ecfp_pca = np.array([list(x) for x in blocks['ecfp_pca']])
        if isinstance(targets, pl.Series):
            self.targets = torch.reshape(torch.from_numpy(dt['binds'].to_numpy()).type(torch.float), (-1, 1))
        else:
            # add dummy targets if this is test. 
            self.targets = torch.from_numpy(np.array([-1]*dt.shape[0])).type(torch.float) 

    def __len__(self):
        return self.blocks.shape[0]
    
    def __getitem__(self, idx):
        idt = self.blocks[idx]
        iblocks_ecfp_pca = [self.blocks_ecfp_pca[idt[x]] for x in ['buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index']]
        iblocks_ecfp_pca = np.concatenate(iblocks_ecfp_pca, axis = 1)
        return self.ids[idx], torch.from_numpy(iblocks_ecfp_pca).type(torch.float), self.targets[idx]

def get_loader(indir, protein_name, n_files = False):
    
    print(f'loading {indir} {protein_name}')
    istest = 'test' in indir
    getcols = ['id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index'] + ([] if istest else ['binds'])
    
    if n_files:
        dt = []
        for file in np.random.choice(listfiles(f'{indir}/base/', protein_name), n_files):
            dt.append(pl.read_parquet(file, columns = getcols))
            del file
        dt = pl.concat(dt)
    else:
        dt = pl.read_parquet(f'{indir}/base/base-{protein_name}-*.parquet', columns = getcols)

    print(f'read {dt.shape[0]/1000/1000:,.2f} M rows')

    blocks_file = 'out/train/building_blocks.parquet' if not istest else 'out/test/building_blocks.parquet'
    blocks = pl.read_parquet(blocks_file, columns = ['index', 'ecfp_pca'])

    if istest:
        targets = None
        batch_size = 1000
        shuffle = False
    else:
        targets = dt['binds']
        batch_size = 100
        shuffle = True
    
    return DataLoader(Dataset_Blocks(dt, blocks, targets), batch_size=batch_size, shuffle=shuffle, num_workers=0)


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