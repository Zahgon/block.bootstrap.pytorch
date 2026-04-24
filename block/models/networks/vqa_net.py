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
        super(VQANet, self).__init__()
        self.self_q_att = self_q_att
        self.wid_to_word = wid_to_word
        self.word_to_wid = word_to_wid
        self.aid_to_ans = aid_to_ans
        self.ans_to_aid = ans_to_aid
        # Modules
        self.txt_enc = factory_text_enc(self.wid_to_word, txt_enc)
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(2400, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.attention = Attention(**attention)
        self.fusion = factory_fusion(classif['fusion'])

    def forward(self, batch):
        pass

    def process_question(self, q, l):
        pass

    def process_answers(self, out):
        pass


class Attention(nn.Module):

    def __init__(self, mlp_glimpses=0, fusion={}):
        super(Attention, self).__init__()
        self.mlp_glimpses = mlp_glimpses
        self.fusion = factory_fusion(fusion)
        if self.mlp_glimpses > 0:
            self.linear0 = nn.Linear(fusion['output_dim'], 512)
            self.linear1 = nn.Linear(512, mlp_glimpses)

    def forward(self, q, v):
        pass

    def process_attention(self, q, v):
        pass
