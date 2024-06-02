# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

import duckdb, torch, platform
import torch.optim as optim
import pandas as pd
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, auc, roc_curve
import numpy as np
from sklearn.metrics import average_precision_score

on_gcp = platform.system() != 'Windows'

data_folder = '' if on_gcp else 'data/'
sample_size = None if on_gcp else 100000
test_samples = None

def datapipe(data_path, sample_size = None):
    
    istest = 'test' in data_path
    data_path = data_folder + data_path
    
    print(f'loading {data_path}')
    
    cols = ['id', 'molecule_smiles'] + (['binds', 'protein_name'] if not istest else [])
        
    if sample_size:
        print(f'sample to {sample_size}')
        df = pl.read_parquet(data_path, columns = cols, n_rows = sample_size)
    else:
        df = pl.read_parquet(data_path, columns = cols)
    
    print(f'read {df.shape[0]} rows')
    
    if not istest:   
    
        df = df.filter(~pl.col('molecule_smiles').is_null())
        
        binds = df['binds'].to_numpy() == 1
        BRD4 = (df['protein_name'] == 'BRD4').to_numpy()
        sEH = (df['protein_name'] == 'sEH').to_numpy()       
        
        binds = np.reshape(binds, (-1, 1))
        BRD4 = np.reshape(BRD4, (-1, 1))
        sEH = np.reshape(sEH, (-1, 1))
        
        targets = np.concatenate((binds, BRD4, sEH), axis = 1)
        
        # target = np.full(len(binds), -1, dtype=np.int8)
        # target[binds & BRD4] = 0 
        # target[binds & sEH] = 1
        # target[binds & ~BRD4 & ~sEH] = 2
        # target[~binds & sEH] = 3
        # target[~binds & BRD4] = 4
        # target[~binds & ~BRD4 & ~sEH] = 5
        # target = np.reshape(target, (-1, 1))
    
    if not istest:
        loader = LeashDataset(df['molecule_smiles'], targets)
        batch_size = 100
        shuffle = True
    else:
        loader = LeashDataset_Test(df['molecule_smiles'])
        batch_size = 100
        shuffle = False
    
    return DataLoader(loader, batch_size=batch_size, shuffle=shuffle, num_workers=0), df['id']

class LeashDataset(Dataset):    
    def __init__(self, smiles, targets):
        self.smiles = smiles
        self.targets = torch.from_numpy(targets).type(torch.float)
    def __len__(self):        
        return len(self.smiles)
    def __getitem__(self, idx):
        if isinstance(idx, int): idx = [idx]
        ecfp = np.array([generate_ecfp(x) for x in self.smiles[idx].to_numpy()])
        return torch.from_numpy(ecfp).type(torch.float), self.targets[idx]

class LeashDataset_Test(Dataset):    
    def __init__(self, smiles):         
        self.smiles = smiles  
    def __len__(self):
        return len(self.smiles)    
    def __getitem__(self, idx):
        if isinstance(idx, int): idx = [idx]
        ecfp = np.array([generate_ecfp(x) for x in self.smiles[idx].to_numpy()])
        return torch.from_numpy(ecfp).type(torch.float)

def generate_ecfp(smile, radius=2, bits=1024):
    molecule = Chem.MolFromSmiles(smile)
    if molecule is None:
        return np.full(bits, -1)
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

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
        self.fc_fin = nn.Linear(10, 3)
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

loader, ids = datapipe('train.parquet', sample_size = sample_size)
net = MLP()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

def unlist(x):
    flatten = lambda *n: (e for a in n
        for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))
    return list(flatten(x))

def print_results(epoch, batch, predictions, labels, scores, loss = None):

    labels = unlist(labels)
    predictions = unlist(predictions)
    scores = unlist(scores)
    
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    auc_result = auc(fpr, tpr)
    gini = auc_result*2-1
    f1 = f1_score(labels, predictions)
    aps = average_precision_score(labels, scores)

    print({
        'epoch': epoch,
        'batch': batch,
        'loss': round(loss, 4) if loss else 'val',
        'f1': round(f1, 4),
        'gini': round(gini, 4),
        'aps': round(aps, 4)
    })
    
    return aps
    
num_epochs = 5

print(f'{len(loader)} batches')
print('training..')

print_batches = 250

for epoch in range(num_epochs):  # loop over the dataset multiple times

    # train loop.
    print(f'epoch {epoch}')

    running_loss = 0.0
    epoch_loss = 0
    running_predictions = []
    running_labels = []
    running_scores = []

    for i, data in enumerate(loader, 0):
    
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)        
        loss = criterion(outputs, labels)
        loss.backward()
        running_loss += loss.item()
        epoch_loss += loss.item()
        del loss
            
        optimizer.step()
        
        running_labels += labels.tolist()
        running_predictions += [1.0 if x[0] > 0.5 else 0.0 for x in outputs.tolist()]
        running_scores += outputs.tolist()

        # print every n batches.
        # if (i % print_batches == 0) and (i != 0):
        #     print_results(
        #         epoch = epoch,
        #         batch = i,
        #         predictions = running_predictions,
        #         labels = running_labels,
        #         scores = running_scores,
        #         loss = running_loss,
        #     )

        #     running_loss = 0.0
        #     running_predictions = []
        #     running_labels = []
        #     running_scores = []
            
        del data, inputs, labels, outputs

loader, ids = datapipe('test.parquet', sample_size = test_samples)

print('running test predictions')
print(f'{len(loader)} batches')
net.eval()
scores = []
with torch.no_grad():
    for i, data in enumerate(loader, 0):
        if (i % 1000 == 0) and (i != 0):
            print(f'batch {i}')
        outputs = net(data)
        scores+= [x[0] for x in outputs.numpy()]

results = pl.DataFrame({'id': ids, 'binds': scores})
results.write_parquet('out/submission.parquet')
    
