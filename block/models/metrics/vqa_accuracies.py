import os
import json
import torch
import torch.nn as nn
import numpy  as np
from scipy import stats
from collections import defaultdict
from bootstrap.lib.logger import Logger
from .vqa_accuracy import VQAAccuracy

class VQAAccuracies(nn.Module):

    def __init__(self,
            engine=None,
            mode='eval',
            open_ended=True,
            tdiuc=True,
            dir_exp='',
            dir_vqa=''):
        raise NotImplementedError

    def reset_oe(self):
        pass

    def save_logits(self):
        pass

    def reset_tdiuc(self):
        pass

    def forward(self, cri_out, net_out, batch):
        pass

    def compute_oe_accuracy(self):
        pass
            # TODO: make it a subprocess call

    def compute_tdiuc_metrics(self):
        pass
