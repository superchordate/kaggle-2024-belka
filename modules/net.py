import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

def unlist(x):
    return np.reshape(x, (1,-1))[0]

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
    
def train(
        loader, 
        save_folder,
        save_name,
        epochs = 5, print_batches = 500,        
        net = MLP(), 
        optimizer = None,
        # criterion = nn.MSELoss()
        criterion = nn.BCELoss()
):    
    if not optimizer: optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    print(f'training {save_name}')
    print(f'{len(loader)} batches')
    for epoch in range(epochs):
        print(f'epoch {epoch}')
        running_loss = 0.0
        epoch_loss = 0
        running_scores = []
        running_labels = []
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)        
            loss = criterion(outputs, labels)
            loss.backward()    
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            running_labels += labels.tolist()
            running_scores += outputs.tolist()
            if (i % print_batches == 0) and (i != 0):
                print(f'batch {i}, loss: {running_loss:.4f}')
                running_loss = 0.0
                save_model(net, save_folder, save_name, verbose = False)
            del i, data, inputs, labels, outputs, loss            
    return net, np.array(unlist(running_labels)), np.array(unlist(running_scores))


    
def run_val(loader, net):    
    print(f'{len(loader)} batches')
    with torch.no_grad():
        for epoch in range(5):
            # print(f'epoch {epoch}')
            running_scores = []
            running_labels = []
            for i, data in enumerate(loader, 0):
                inputs, labels = data
                outputs = net(inputs)
                running_labels += labels.tolist()
                running_scores += outputs.tolist()
                del i, data, inputs, labels, outputs
    return np.array(unlist(running_labels)), np.array(unlist(running_scores))

def save_model(model, folder, name, on_gcp = False, verbose = True, device = 'cpu'):
    model_scripted = torch.jit.script(model.cpu())
    model_scripted.save(f'{folder}/{name}.pt')
    if on_gcp: os.system(f'gsutil cp {name}.pt gs://kaggle-417721/{name}.pt')
    if verbose: print(f'saved {folder}/{name}.pt')
    model = model.to(device)
    model = model.train()