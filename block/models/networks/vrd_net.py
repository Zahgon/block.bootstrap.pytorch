import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .fusions.factory import factory as factory_fusion
from .mlp import MLP

class VRDNet(nn.Module):

    def __init__(self, opt):
        super(VRDNet, self).__init__()
        self.opt = opt
        self.classeme_embedding = nn.Embedding(
            self.opt['nb_classeme'],
            self.opt['classeme_dim'])
        self.fusion_c = factory_fusion(self.opt['classeme'])
        self.fusion_s = factory_fusion(self.opt['spatial'])
        self.fusion_f = factory_fusion(self.opt['feature'])
        self.predictor = MLP(**self.opt['predictor'])

    def forward(self, batch):
        pass
