import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.init import normal_
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
    
class Siamese(nn.Module):

    def __init__(self, options, input_len):

        super(Siamese, self).__init__()    

        self.dropoutpct = options['dropout']/100
        self.input_len = input_len
        print(f'input_len: {input_len}') 

        self.siamese = nn.Sequential(
            nn.Linear(self.input_len, self.input_len),
            nn.BatchNorm1d(self.input_len),
            KAF(1, conv = False),
            nn.Dropout(self.dropoutpct),
            nn.Linear(self.input_len, 1000),
            nn.BatchNorm1d(1000),
            KAF(1, conv = False),
            nn.Dropout(self.dropoutpct),
            nn.Linear(1000, 100),
            KAF(1, conv = False),
        )

        self.out = nn.Linear(100, 1)

    def forward_once(self, x):
        x = self.siamese(x)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2, x3):
        # block1 = self.forward_one(x1)
        # block2 = self.forward_one(x2)
        # block3 = self.forward_one(x3)
        # return self.out(torch.abs(
        #     block1*(block2 - block3) + block2*(block3 - block1) + block3*(block1 - block2)
        # ))
        return self.forward_once(x1), self.forward_once(x2), self.forward_once(x3)

# Define the Contrastive Loss Function
# https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/#siamese-neural-networks-in-pytorch

# https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/#siamese-neural-networks-in-pytorch
class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, output3, label):
      
      # Calculate the euclidean distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True) + \
          F.pairwise_distance(output1, output3, keepdim = True) + \
          F.pairwise_distance(output2, output3, keepdim = True)

      loss_contrastive = torch.mean(
          (1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

      return loss_contrastive
    
class KAF(nn.Module):

    """ Implementation of the kernel activation function.
    
    Parameters
    ----------
    num_parameters: int
        Size of the layer (number of neurons).
    D: int, optional
        Size of the dictionary for each neuron. Default to 20.
    conv: bool, optional
        True if this is a convolutive layer, False for a feedforward layer. Default to False.
    boundary: float, optional
        Dictionary elements are sampled uniformly in [-boundary, boundary]. Default to 4.0.
    init_fcn: None or func, optional
        If None, elements are initialized randomly. Otherwise, elements are initialized to approximate given function.
    kernel: {'gauss', 'relu', 'softplus'}, optional
        Kernel function to be used. Defaults to 'gaussian'.
    
    Example
    ----------
    Neural network with one hidden layer with KAF nonlinearities:
        
    >>> net = Sequential([nn.Linear(10, 20), KAF(20), nn.Linear(20, 1)])
    
    References
    ----------
    [1] Scardapane, S., Van Vaerenbergh, S., Totaro, S. and Uncini, A., 2019. 
        Kafnets: kernel-based non-parametric activation functions for neural networks. 
        Neural Networks, 110, pp. 19-32.
    [2] Marra, G., Zanca, D., Betti, A. and Gori, M., 2018. 
        Learning Neuron Non-Linearities with Kernel-Based Deep Neural Networks. 
        arXiv preprint arXiv:1807.06302.
    """

    def __init__(self, num_parameters, D=20, conv=False, boundary=4.0, init_fcn=None, kernel='gaussian'):

        super().__init__()
        self.num_parameters, self.D, self.conv = num_parameters, D, conv
        
        # Initialize the dictionary (NumPy)
        self.dict_numpy = np.linspace(-boundary, boundary, self.D).astype(np.float32).reshape(-1, 1)
        
        # Save the dictionary
        if self.conv:
            self.register_buffer('dict', torch.from_numpy(self.dict_numpy).view(1, 1, 1, 1, -1))
            self.unsqueeze_dim = 4
        else:
            self.register_buffer('dict', torch.from_numpy(self.dict_numpy).view(1, -1))
            self.unsqueeze_dim = 2

        # Select appropriate kernel function
        if not (kernel in ['gaussian', 'relu', 'softplus']):
            raise ValueError('Kernel not recognized (must be {gaussian, relu, softplus})')
            
        if kernel == 'gaussian':
          self.kernel_fcn = self.gaussian_kernel
          # Rule of thumb for gamma (only needed for Gaussian kernel)
          interval = (self.dict_numpy[1] - self.dict_numpy[0])
          sigma = 2 * interval  # empirically chosen
          self.gamma_init = float(0.5 / np.square(sigma))
          
          # Initialize gamma
          if self.conv:
              self.register_buffer('gamma', torch.from_numpy(np.ones((1, 1, 1, 1, self.D), dtype=np.float32)*self.gamma_init))
          else:
              self.register_buffer('gamma', torch.from_numpy(np.ones((1, 1, self.D), dtype=np.float32)*self.gamma_init))
          
        elif kernel == 'relu':
          self.kernel_fcn = self.relu_kernel
        else:
          self.kernel_fcn = self.softplus_kernel

        # Initialize mixing coefficients
        if self.conv:
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, 1, 1, self.D))
        else:
            self.alpha = Parameter(torch.FloatTensor(1, self.num_parameters, self.D))
        
        # Eventually: initialization with kernel ridge regression
        self.init_fcn = init_fcn
        if init_fcn != None:
            
            if kernel == 'gaussian':
              K = np.exp(- self.gamma_init*(self.dict_numpy - self.dict_numpy.T) ** 2)
            elif kernel == 'softplus':
              K = np.log(np.exp(self.dict_numpy - self.dict_numpy.T) + 1.0)
            else:
              #K = np.maximum(self.dict_numpy - self.dict_numpy.T, 0)
              raise ValueError('Cannot perform kernel ridge regression with ReLU kernel (singular matrix)')
            
            self.alpha_init = np.linalg.solve(K + 1e-4 * np.eye(self.D), self.init_fcn(self.dict_numpy)).reshape(-1).astype(np.float32)
            
        else:   
            self.alpha_init = None
        
        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_fcn != None:
            if self.conv:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1, 1, 1)
            else:
                self.alpha.data = torch.from_numpy(self.alpha_init).repeat(1, self.num_parameters, 1)
        else:
            normal_(self.alpha.data, std=0.8)
    
    def gaussian_kernel(self, input):
      return torch.exp(- torch.mul((torch.add(input.unsqueeze(self.unsqueeze_dim), - self.dict))**2, self.gamma))
    
    def relu_kernel(self, input):
      return F.relu(input.unsqueeze(self.unsqueeze_dim) - self.dict)
    
    def softplus_kernel(self, input):
        return F.softplus(input.unsqueeze(self.unsqueeze_dim) - self.dict)
    
    def forward(self, input):
        K = self.kernel_fcn(input)
        y = torch.sum(K*self.alpha, self.unsqueeze_dim)
        return y

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.num_parameters) + ')'
    
