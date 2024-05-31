# https://www.kaggle.com/code/jetakow/home-credit-2024-starter-notebook/notebook

import torch, platform, os
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

on_gcp = platform.system() != 'Windows'
os.system("gsutil cp gs://kaggle-417721/train.parquet train.parquet")
os.system("gsutil cp gs://kaggle-417721/test.parquet test.parquet")

data_folder = 'data/' if on_gcp else ''
sample_size = None if on_gcp else 100000

def generate_ecfp(molecule, radius=2, bits=1024):
    if molecule is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(molecule, radius, nBits=bits))

class LeashDataset(Dataset):
    
    def __init__(self, ecfp, targets): 
        
        self.ecfp = torch.from_numpy(ecfp).type(torch.float)
        self.targets = torch.from_numpy(targets).type(torch.float)
        
    def __len__(self):        
        return len(self.ecfp)
    
    def __getitem__(self, idx):
        if len(self.targets) > 0:
            return self.ecfp[idx], self.targets[idx]
        else:
            return self.ecfp[idx]

def datapipe(data_path, sample_size = None, istest = False):
    
    data_path = data_folder + data_path
    
    print(f'loading {data_path}')
    
    cols = ['id', 'molecule_smiles']
    if not istest:
        cols.append('binds')
    if sample_size:
        print(f'sample to {sample_size}')
        df = pl.read_parquet(data_path, columns = ['id', 'molecule_smiles'], n_rows = sample_size)
    else:
        df = pl.read_parquet(data_path, columns = ['id', 'molecule_smiles'])
    
    df = df.to_pandas()
    df['molecule'] = df['molecule_smiles'].apply(Chem.MolFromSmiles)
    df['ecfp'] = df['molecule'].apply(generate_ecfp)

    ecfp = np.array([list(x) for x in df['ecfp']], dtype=np.float32)
    if not istest:
        targets = np.reshape(np.array(df['binds'].values, dtype=np.float32), (-1, 1))
    else:
        targets = []
    
    return DataLoader(LeashDataset(ecfp, targets), batch_size=100, shuffle=True, num_workers=0), df['id']


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

loader, ids = datapipe('data/train.parquet', sample_size = sample_size)
net = MLP()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

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

    print({
        'epoch': epoch,
        'batch': batch,
        'loss': round(loss, 4) if loss else 'val',
        'f1': round(f1, 4),
        'gini': round(gini, 4)
    })
    
num_epochs = 5

print('training..')

print_batches = 100

for epoch in range(num_epochs):  # loop over the dataset multiple times

    # train loop.

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
        if (i % print_batches == 0) and (i != 0):
            print_results(
                epoch = epoch,
                batch = i,
                predictions = running_predictions,
                labels = running_labels,
                scores = running_scores,
                loss = running_loss,
            )

            running_loss = 0.0
            running_predictions = []
            running_labels = []
            running_scores = []
            
        del data, inputs, labels, outputs

loader, ids = datapipe('data/test.parquet')

print('running test predictions')
net.eval()
scores = []
for i, data in enumerate(loader, 0):

    inputs, labels = data
    outputs = net(inputs)
    scores+= outputs.to_list()

results = pd.DataFrame({'id': ids, 'binds': unlist(scores)})

if not os.path.exists('out'):
    os.makedirs('out')
results.to_parquet('out/submission.parquet', index = False)
    
os.system("gsutil cp out/submission.parquet gs://kaggle-417721/submission.parquet")