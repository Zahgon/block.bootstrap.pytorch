import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.metrics.accuracy import accuracy
from ..criterions.vrd_bce import VRDBCELoss
from . import vrd_utils

class VRDPredicate(nn.Module):

    def __init__(self, engine=None, split='test', nb_classes=71):
        raise NotImplementedError

    def reset(self):
        pass

    def forward(self, cri_out, net_out, batch):
        pass

    def calculate_metrics(self):
        pass
