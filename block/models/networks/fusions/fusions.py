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
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size*rank)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size*rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(BlockTucker, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert(pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if self.shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)

        self.sizes_list = get_sizes_list(mm_dim, chunks)
        bilinears = []
        for size in self.sizes_list:
            bilinears.append(
                nn.Bilinear(size, size, size)
            )
        self.bilinears = nn.ModuleList(bilinears)
        self.linear_out = nn.Linear(self.mm_dim, self.output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(Mutan, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.rank = rank
        self.output_dim = output_dim
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        self.normalize = normalize
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.merge_linear0 = nn.Linear(mm_dim, mm_dim*rank)
        if self.shared:
            self.linear1 = self.linear0
            self.merge_linear1 = self.merge_linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
            self.merge_linear1 = nn.Linear(mm_dim, mm_dim*rank)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(Tucker, self).__init__()
        self.input_dims = input_dims
        self.shared = shared
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.bilinear = nn.Bilinear(mm_dim, mm_dim, mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(MLB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(MFB, self).__init__()
        self.input_dims = input_dims
        self.mm_dim = mm_dim
        self.factor = factor
        self.output_dim = output_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(MFH, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0_0 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_0 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear0_1 = nn.Linear(input_dims[0], mm_dim*factor)
        self.linear1_1 = nn.Linear(input_dims[1], mm_dim*factor)
        self.linear_out = nn.Linear(mm_dim*2, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        pass


class MCB(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            mm_dim=16000,
            activ_output='relu',
            dropout_output=0.):
        super(MCB, self).__init__()
        # compatible with pytorch 0.3 and 0.4, not 1.0
        from . import compactbilinearpooling as cbp
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_output = activ_output
        self.dropout_output = dropout_output
        # Modules
        self.mcb = cbp.CompactBilinearPooling(input_dims[0], input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        super(LinearSum, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        # Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        self.linear1 = nn.Linear(input_dims[1], mm_dim)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        pass


class ConcatMLP(nn.Module):

    def __init__(self,
            input_dims,
            output_dim,
            dimensions=[500,500],
            activation='relu',
            dropout=0.):
        super(ConcatMLP, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.input_dim = sum(input_dims)
        self.dimensions = dimensions + [output_dim]
        self.activation = activation
        self.dropout = dropout
        # Modules
        self.mlp = mlp.MLP(
            self.input_dim,
            self.dimensions,
            self.activation,
            self.dropout)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        pass
