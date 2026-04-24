import sys
import copy
import torch
import torch.nn as nn
from bootstrap.lib.options import Options
from bootstrap.models.networks.data_parallel import DataParallel
from .vqa_net import VQANet
from .vrd_net import VRDNet

def factory(engine):
    pass
