import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import mlp

def get_sizes_list(dim, chunks):
    pass

def get_chunks(x,sizes):
    pass


class Block(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            rank=15,
            shared=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        raise NotImplementedError

    def forward(self, x):
        pass


class BlockTucker(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            chunks=20,
            shared=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.,
            pos_norm='before_cat'):
        raise NotImplementedError

    def forward(self, x):
        pass


class Mutan(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            rank=15,
            shared=False,
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class Tucker(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1600,
            shared=False,
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class MLB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class MFB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            factor=2,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_norm=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class MFH(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            factor=2,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class MCB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=16000,
            activ_output='relu',
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class LinearSum(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=1200,
            activ_input='relu',
            activ_output='relu',
            normalize=False,
            dropout_input=0.,
            dropout_pre_lin=0.,
            dropout_output=0.):
        raise NotImplementedError

    def forward(self, x):
        pass


class ConcatMLP(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            dimensions=[500,500],
            activation='relu',
            dropout=0.):
        raise NotImplementedError

    def forward(self, x):
        pass
