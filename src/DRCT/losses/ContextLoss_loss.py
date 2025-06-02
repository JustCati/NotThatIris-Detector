import torch.nn as nn
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class ContextLoss(nn.Module):
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        super(ContextLoss, self).__init__()

    def forward(self, input, target):
        return nn.MSELoss(reduction=self.reduction).to(input.device)(input, target)
