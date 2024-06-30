import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_sm(nn.Module):
    def __init__(self, options, input_len = 1024):

        super().__init__()

        self.dropoutpct = options['dropout']/100
        self.input_len = input_len
        print(f'input_len: {input_len}')

        self.fc1 = nn.Linear(self.input_len, 1000)
        self.batchnorm1 = nn.BatchNorm1d(1000)
        self.dropout1 = nn.Dropout(self.dropoutpct)

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

class MLP_md(nn.Module):

    def __init__(self, options, input_len = 1024):

        super().__init__()

        self.dropoutpct = options['dropout']/100
        self.input_len = input_len
        print(f'input_len: {input_len}')

        self.fc1 = nn.Linear(self.input_len, self.input_len)
        self.batchnorm1 = nn.BatchNorm1d(self.input_len )

        self.fc3 = nn.Linear(self.input_len, 1000)
        self.batchnorm3 = nn.BatchNorm1d(1000)
        self.dropout3 = nn.Dropout(self.dropoutpct)

        self.fc4 = nn.Linear(1000, 100)
        self.batchnorm4 = nn.BatchNorm1d(100)
        self.dropout4 = nn.Dropout(self.dropoutpct)

        self.fc5 = nn.Linear(100, 10)
        self.fc_fin = nn.Linear(10, 3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.fc1(x))
        x = self.batchnorm1(x)

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
    
class MLP_lg(nn.Module):

    def __init__(self, options, input_len = 1024):

        super().__init__()

        self.dropoutpct = options['dropout']/100
        self.input_len = input_len
        print(f'input_len: {input_len}')

        self.fc1 = nn.Linear(self.input_len, self.input_len * 3)
        self.batchnorm1 = nn.BatchNorm1d(self.input_len * 3)

        self.fc2 = nn.Linear(self.input_len * 3, self.input_len)
        self.batchnorm2 = nn.BatchNorm1d(self.input_len)
        self.dropout2 = nn.Dropout(self.dropoutpct)

        self.fc3 = nn.Linear(self.input_len, 500)
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

        x = F.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)

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