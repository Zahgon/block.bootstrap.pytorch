import torch
import torch.nn as nn
from bootstrap.models.metrics.accuracy import accuracy

class VQAAccuracy(nn.Module):

    def __init__(self, topk=[1,5]):
        raise NotImplementedError

    def __call__(self, cri_out, net_out, batch):
        raise NotImplementedError
