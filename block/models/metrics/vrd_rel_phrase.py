import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from bootstrap.models.metrics.accuracy import accuracy
from . import vrd_utils

class VRDRelationshipPhrase(nn.Module):

    def __init__(self, engine=None, split='test'):
        raise NotImplementedError

    def reset(self):
        # Relationship task metrics
        pass

    def forward(self, cri_out, net_out, batch):
        pass

    def calculate_metrics(self):
        pass
