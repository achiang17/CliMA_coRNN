import torch
import torch.nn as nn

class NKGELoss(nn.Module):
    def __init__(self):
        super(NKGELoss, self).__init__()

    def forward(self, predictions, targets):
        # Ensure the predictions and targets are of the same shape
        if predictions.size() != targets.size():
            raise ValueError("Predictions and targets must have the same shape")

        # # alpha : variability of prediciton errors
        # pred_var = (torch.std(predictions))**2
        # target_var = (torch.std(targets))**2
        # alpha = pred_var / target_var

        # gamma 
        gamma = (torch.std(predictions)/torch.mean(predictions))/(torch.std(targets)/torch.mean(targets))

        # beta : bias term
        pred_mean = torch.mean(predictions)
        target_mean = torch.mean(targets)
        beta = pred_mean / target_mean
        
        # r : Pearson correlation coefficient
        cov = torch.mean((predictions - pred_mean) * (targets - target_mean))
        pred_std = torch.std(predictions)
        target_std = torch.std(targets)
        r = cov / (pred_std * target_std)

        # Calculate KGE
        kge = 1 - torch.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        #nkge = 1 / (2 - kge)

        # Maximize nkge
        return 1 - kge
