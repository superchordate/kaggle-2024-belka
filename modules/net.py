import torch, os, time, gc
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import polars as pl
from modules.utils import device, gcp
from modules.features import features
from modules.datasets import get_loader

class MLP(nn.Module):
    def __init__(self, input_len = 1024):
        super().__init__()
        self.input_len = input_len
        self.fc1 = nn.Linear(self.input_len, self.input_len)
        self.batchnorm1 = nn.BatchNorm1d(self.input_len)
        self.dropout1 = nn.Dropout(0.25)        
        self.fc3 = nn.Linear(self.input_len, 500)
        self.batchnorm3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(500, 100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.dropout4 = nn.Dropout(0.3)
        self.fc5 = nn.Linear(100, 10)
        self.fc_fin = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch        
        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.dropout1(x)        
        x = F.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.dropout3(x)        
        x = F.relu(self.fc4(x))
        x = self.batchnorm4(x)
        x = self.dropout4(x)        
        x = F.relu(self.fc5(x))        
        x = self.fc_fin(x)
        x = self.sigmoid(x)
        return x

class MLP_multi(nn.Module):
    def __init__(self, options, input_len = 1024):

        super().__init__()

        self.dropoutpct = options['dropout']/100
        self.input_len = input_len
        print(f'input_len: {input_len}')

        self.fc1 = nn.Linear(self.input_len, 1000)
        self.batchnorm1 = nn.BatchNorm1d(1000)
        self.dropout1 = nn.Dropout(self.dropoutpct)

        # self.fc2 = nn.Linear(self.input_len, self.input_len)
        # self.batchnorm2 = nn.BatchNorm1d(self.input_len)
        # self.dropout2 = nn.Dropout(self.dropoutpct)

        self.fc3 = nn.Linear(1000, 500)
        self.batchnorm3 = nn.BatchNorm1d(500)
        self.dropout3 = nn.Dropout(self.dropoutpct)

        self.fc4 = nn.Linear(500, 100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.dropout4 = nn.Dropout(self.dropoutpct)

        self.fc5 = nn.Linear(100, 10)
        self.fc_fin = nn.Linear(10, 3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)
        #x = self.dropout1(x)

        # x = F.relu(self.fc2(x))
        # x = self.batchnorm2(x)
        #x = self.dropout2(x)

        x = F.relu(self.fc3(x))
        x = self.batchnorm3(x)
        x = self.dropout3(x)

        x = F.relu(self.fc4(x))
        x = self.batchnorm4(x)
        x = self.dropout4(x)

        x = F.relu(self.fc5(x))
        x = self.fc_fin(x)
        x = self.sigmoid(x)
        return {
            'sEH': torch.reshape(x[:,0], (-1, 1)), 
            'BRD4': torch.reshape(x[:,1], (-1, 1)), 
            'HSA': torch.reshape(x[:,2], (-1, 1))
        }
    
def train(
        indir, 
        save_folder,
        save_name,        
        options,
        print_batches = 2000,
        net = None, 
        optimizer = None,
        # criterion = nn.MSELoss()
        criterion = None
):  
    idevice = device()

    # load mols and blocks.
    molpath = f'{indir}/mols.parquet'
    print(f'loading {molpath}')
    mols = pl.read_parquet(
        molpath, 
        columns = ['molecule_id', 'buildingblock1_index', 'buildingblock2_index', 'buildingblock3_index', 'binds_sEH', 'binds_BRD4', 'binds_HSA']
    )
    blocks = pl.read_parquet('out/train/blocks/blocks-3-pca.parquet', columns = ['index', 'features_pca'])

    # get the network, optimizer, and criterion.
    if not net:
        print('starting from clean network')
        input_len = len(features(mols[0,], blocks, options)[0])
        net = MLP_multi(options = options, input_len = input_len).to(idevice)
        del input_len
    else:
        print('using existing network')
    
    if not optimizer:
        optimizer = optim.SGD(net.parameters(), lr=options['lr'], momentum=options['momentum'])

    if not criterion: 
        criterion1 = nn.BCELoss().to(idevice)
        criterion2 = nn.BCELoss().to(idevice)
        criterion3 = nn.BCELoss().to(idevice)
        #criterion = nn.CrossEntropyLoss().to(idevice)

    # the data is too large to fit in memory, so we need to load it in batches.
    if options['n_rows'] == 'all':
        num_splits = 150
        mols = mols.with_columns(pl.Series('group', np.random.choice(range(num_splits), mols.shape[0])))
        mols = mols.partition_by('group', include_key = False)
        print(f'split mols to {num_splits} random splits for processing.')
    else:
        mols = [mols.sample(options['n_rows'])]
        print(f'sampled to {options["n_rows"]/1000/1000:.1f}M rows.')
    
    print(f'training {save_name}')
    for epoch in range(options['epochs']):

        molct = 0
        for imols in mols:

            molct += 1
            print(f'epoch {epoch + 1} split {molct} of {num_splits}')
            
            loader = get_loader(indir = '', mols = imols, blocks = blocks, options = options)
            print(f'{len(loader):,.0f} batches')
            
            start_time = time.time()
            loss = 0.0
            scores = {'sEH': [], 'BRD4': [], 'HSA': []}
            labels = {'sEH': [], 'BRD4': [], 'HSA': []}
            for i, data in enumerate(loader, 0):
                
                imolecule_ids, iX, iy = data
                optimizer.zero_grad()
                outputs = net(iX)
                
                loss1 = criterion1(outputs['sEH'], iy['sEH'])
                loss2 = criterion2(outputs['BRD4'], iy['BRD4'])
                loss3 = criterion3(outputs['HSA'], iy['HSA'])
                
                iloss = loss1 + loss2 + loss3
                iloss.backward()
                optimizer.step()

                loss += iloss.cpu().item()
                for protein_name in iy.keys():
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                    scores[protein_name] = np.append(scores[protein_name], outputs[protein_name].cpu().tolist())

                if (i % print_batches == 0) and (i != 0):
                    print(f'batch {i}, loss: {loss:.0f} {(time.time() - start_time)/60:.1f} mins')
                    start_time = time.time()
                    loss = 0.0
                    save_model(net, save_folder, save_name, verbose = False)

                del i, data, imolecule_ids, iX, iy, outputs, loss1, loss2, loss3, iloss
            
            del imols, loader
            gc.collect()
        
        save_model(net, save_folder, save_name, verbose = False)
        return net, labels, scores
    
def run_val(loader, net, print_batches = 2000): 

    print(f'{len(loader)} batches')

    net = net.eval()

    with torch.no_grad():
        scores = {'sEH': [], 'BRD4': [], 'HSA': []}
        labels = {'sEH': [], 'BRD4': [], 'HSA': []}
        molecule_ids = []
        for i, data in enumerate(loader, 0):
            
            imolecule_ids, iX, iy = data
            outputs = net(iX)
            
            for protein_name in outputs.keys():
                if len(iy[protein_name]) > 0: # will be empty if this is a test set.
                    labels[protein_name] = np.append(labels[protein_name], iy[protein_name].cpu().tolist())
                scores[protein_name] = np.append(scores[protein_name], outputs[protein_name].cpu().tolist())

            molecule_ids = np.append(molecule_ids, imolecule_ids)
            if (i % print_batches == 0) and (i != 0):
                print(f'batch {i}')
            del i, data, imolecule_ids, iX, iy, outputs
            
    return molecule_ids, labels, scores

def save_model(model, folder, name, verbose = True):
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f'{folder}/{name}.pt')
    if gcp(): os.system(f'gsutil cp {folder}/{name}.pt gs://kaggle-417721/{name}.pt')
    if verbose: print(f'saved {folder}/{name}.pt')
    model = model.to(device())
    model = model.train()

        