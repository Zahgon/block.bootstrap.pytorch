import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import skipthoughts
from bootstrap.datasets import transforms as bootstrap_tf
from bootstrap.lib.options import Options
from .fusions.factory import factory as factory_fusion

def mask_softmax(x, lengths):#, dim=1)
    pass

def factory_text_enc(vocab_words, opt):
    pass


class VQANet(nn.Module):

    def __init__(self,
            txt_enc={},
            self_q_att=False,
            attention={},
            classif={},
            wid_to_word={},
            word_to_wid={},
            aid_to_ans=[],
            ans_to_aid={}):
        raise NotImplementedError

    def forward(self, batch):
        pass

    def process_question(self, q, l):
        pass

    def process_answers(self, out):
        pass


class Attention(nn.Module):

    def __init__(self, mlp_glimpses=0, fusion={}):
        raise NotImplementedError

    def forward(self, q, v):
        pass

    def process_attention(self, q, v):
        pass
