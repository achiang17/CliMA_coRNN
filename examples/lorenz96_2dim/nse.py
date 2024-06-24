import torch
from torch import nn

class NSELoss(nn.Module):
    def __init__(self):
        super(NSELoss, self).__init__()
    
    def forward(self, predictions, targets):
        squared_error = (targets - predictions) ** 2
        eps = 0.1
        weights = 1 / (torch.std(targets) + eps) ** 2
        scaled_loss = squared_error * weights
        return torch.mean(scaled_loss)