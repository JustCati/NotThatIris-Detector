import torch
import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY



@LOSS_REGISTRY.register()
class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target):
        mask = target != 0
        return torch.abs(input - target)[mask].mean()
