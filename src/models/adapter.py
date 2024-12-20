import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F




class Adapter(pl.LightningModule):
    def __init__(self, in_features, out_features, verbose=False):
        super().__init__()
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=1))
        self.Adapter = nn.Linear(in_features, out_features)
        if verbose:
            print(self.Adapter)


    def forward(self, x):
        return self.Adapter(x)


    def triplet_accuracy(self, anchor, positive, negative, epsilon=1e-6):
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        similarity_positive = F.cosine_similarity(anchor, positive, dim=1)
        similarity_negative = F.cosine_similarity(anchor, negative, dim=1)

        accuracy = (similarity_positive + epsilon > similarity_negative).float().mean()
        return accuracy.item()


    def training_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

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

        triplet_accuracy = self.triplet_accuracy(anchor, positive, negative)
        self.log("eval/triplet_accuracy", triplet_accuracy)


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
