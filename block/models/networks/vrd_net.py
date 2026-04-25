import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .fusions.factory import factory as factory_fusion
from .mlp import MLP

class VRDNet(nn.Module):

    def __init__(self, opt):
        raise NotImplementedError

    def forward(self, batch):
        pass
