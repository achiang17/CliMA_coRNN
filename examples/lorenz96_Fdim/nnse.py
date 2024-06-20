import torch
from torch import nn

class NNSELoss(nn.Module):
    def __init__(self):
        super(NNSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        numerator = torch.sum((targets - predictions) ** 2)
        denominator = torch.sum((targets - torch.mean(targets)) ** 2)
        nse = 1 - (numerator / denominator)
        nnse = 1/(2-nse)
        return 1-nnse