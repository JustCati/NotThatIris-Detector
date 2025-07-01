import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY



@LOSS_REGISTRY.register()
class WeightedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target):
        mask = target != 0
        value = torch.abs(input - target)[mask]
        
        if self.reduction == 'mean':
            return value.mean()
        elif self.reduction == 'sum':
            return value.sum()
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
