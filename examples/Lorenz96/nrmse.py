import torch
from torch import nn

class NRMSELoss(nn.Module):
    def __init__(self):
        super(NRMSELoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        mse_loss = torch.mean((y_true - y_pred) ** 2)
        rmse = torch.sqrt(mse_loss)
        rms_true = torch.sqrt(torch.mean(y_true ** 2))
        nrmse_value = rmse / rms_true
        return nrmse_value
