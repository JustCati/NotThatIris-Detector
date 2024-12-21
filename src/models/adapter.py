import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F
from collections import OrderedDict




class Adapter(pl.LightningModule):
    def __init__(self, in_features, margin=1, verbose=False):
        super().__init__()
        self.margin = 1
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=1), margin=margin)
        self.Adapter = nn.Linear(in_features, in_features)
        if verbose:
            print(self.Adapter)


    def forward(self, x):
        return self.Adapter(x)


    def training_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        loss = self.criterion(anchor, positive, negative)
        self.log("train/train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

        loss = self.criterion(anchor, positive, negative)
        self.log("eval/val_loss", loss)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
