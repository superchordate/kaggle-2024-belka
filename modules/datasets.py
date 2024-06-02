from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from modules.mols import ecfp
from modules.utils import listfiles
import numpy as np
import polars as pl

class LeashDataset(Dataset):    
    def __init__(self, dt, targets):
        self.dt = dt
        self.targets = torch.from_numpy(targets).type(torch.float)
    def __len__(self):        
        return self.dt.shape[0]
    def __getitem__(self, idx):
        idt = self.dt[idx].map_rows(lambda row: (ecfp(row[0]), ))
        iecfp = np.array([x for x in idt['column_0']])
        return torch.from_numpy(iecfp).type(torch.float), self.targets[idx]

class LeashDataset_Test(Dataset):
    def __init__(self, dt):
        self.dt = dt
    def __len__(self):        
        return self.dt.shape[0]
    def __getitem__(self, idx):
        idt = self.dt[idx].map_rows(lambda row: (ecfp(row[0]), ))
        iecfp = np.array([x for x in idt['column_0']])
        return torch.from_numpy(iecfp).type(torch.float)

def get_loader_multi(indir):
    
    print(f'loading {indir}')
    
    # get base data.
    dt = []
    for protein in ['BRD4', 'HSA', 'sEH']:
        idt = pl.read_parquet(f'{indir}base-sample-{protein}.parquet', columns = ['molecule_smiles', 'binds']).to_pandas()
        idt['protein_name'] = protein
        idt = pl.DataFrame(idt)
        dt.append(idt)
        del idt, protein
    dt = pl.concat(dt)
    dt = dt.select(['protein_name', 'binds', 'molecule_smiles'])    
    print(f'read {dt.shape[0]} rows')
    
    # extract multi-task targets.
    print('extract targets')
    binds = dt['binds'].to_numpy() == 1
    BRD4 = (dt['protein_name'] == 'BRD4').to_numpy()
    sEH = (dt['protein_name'] == 'sEH').to_numpy()
    binds = np.reshape(binds, (-1, 1))
    BRD4 = np.reshape(BRD4, (-1, 1))
    sEH = np.reshape(sEH, (-1, 1))        
    targets = np.concatenate((binds, BRD4, sEH), axis = 1)
    del BRD4, sEH, binds
    
    # build loaders.    
    loader = LeashDataset(dt.select('molecule_smiles'), targets)
    batch_size = 100
    shuffle = True
    
    return DataLoader(loader, batch_size=batch_size, shuffle=shuffle, num_workers=0), dt['protein_name']

def get_loader(indir, protein_name, sample = False):
    
    print(f'loading {indir}')
    istest = 'test' in indir
    getcols = ['molecule_smiles'] if istest else ['binds', 'molecule_smiles']
    
    if sample:
        file = np.random.choice(listfiles(f'{indir}/base/', protein_name), 1)
        dt = pl.read_parquet(file, columns = getcols)
        del file
    else:
        dt = pl.concat([
            pl.read_parquet(file, columns = getcols) \
                for file in listfiles(f'{indir}/base/', protein_name)
        ])
            
    # dt = dt.with_columns(pl.Series(name="protein_name", values=[protein_name]*dt.shape[0]))
    dt = dt.select(getcols)
    print(f'read {dt.shape[0]} rows')

    if istest:
        loader = LeashDataset_Test(dt)
        batch_size = 1000
        shuffle = False
    else:
        targets = np.reshape(dt['binds'], (-1, 1))    
        loader = LeashDataset(dt.select('molecule_smiles'), targets)
        batch_size = 100
        shuffle = True
    
    return DataLoader(loader, batch_size=batch_size, shuffle=shuffle, num_workers=0)

