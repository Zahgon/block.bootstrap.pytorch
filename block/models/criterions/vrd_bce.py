import torch.nn as nn

class VRDBCELoss(nn.Module):

    def __init__(self):
        super(VRDBCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, net_output, target):
        pass
