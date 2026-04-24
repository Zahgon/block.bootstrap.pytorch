import torch.nn as nn

class VQACrossEntropyLoss(nn.Module):

    def __init__(self):
        super(VQACrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, net_out, batch):
        pass
