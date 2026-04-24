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
        super(VQAAccuracies, self).__init__()
        self.engine = engine
        self.mode = mode
        self.open_ended = open_ended
        self.tdiuc = tdiuc
        self.dir_exp = dir_exp
        self.dir_vqa = dir_vqa
        self.dataset = engine.dataset[mode]
        self.ans_to_aid = self.dataset.ans_to_aid
        self.results = None
        self.results_testdev = None
        self.dir_rslt = None
        self.path_rslt = None

        # Module
        if self.tdiuc or self.dataset.split != 'test':
            self.accuracy = VQAAccuracy()
        else:
            self.accuracy = None

        if self.open_ended:
            engine.register_hook(
                '{}_on_start_epoch'.format(mode),
                self.reset_oe)
            engine.register_hook(
                '{}_on_end_epoch'.format(mode),
                self.compute_oe_accuracy)

            # if self.dataset.split == 'test':
            #     engine.register_hook(
            #         '{}_on_end_epoch'.format(mode),
            #         self.save_logits)

        if self.tdiuc:
            engine.register_hook(
                '{}_on_start_epoch'.format(mode),
                self.reset_tdiuc)
            engine.register_hook(
                '{}_on_end_epoch'.format(mode),
                self.compute_tdiuc_metrics)

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
