import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

class MLP(nn.Module):
    
    def __init__(self,
            input_dim,
            dimensions,
            activation='relu',
            dropout=0.):
        raise NotImplementedError
    
    def forward(self, x):
        pass
