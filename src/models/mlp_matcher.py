import torch
import torch.nn as nn
import lightning as pl
from src.engine.thresholding import get_eer



class MLPMatcher(pl.LightningModule):
    def __init__(self, in_feature, num_classes, threshold=None, extractor=None, verbose=False):
        super().__init__()
        self.threshold = threshold
        self.extractor = extractor
        self.Classifier = nn.Linear(in_feature, num_classes)
        if verbose:
            print(self.Classifier)


    def set_threshold(self, threshold):
        self.threshold = threshold


    def forward(self, x):
        if self.extractor is not None:
            x = self.extractor(x)
        x = self.Classifier(x)
        if self.threshold is not None:
            x = x > self.threshold
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1).long()
        y_hat = self(x.to(self.device))

        loss = nn.CrossEntropyLoss()(y_hat, y)
        self.log("train/train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))

        y_for_loss = y.view(-1).long()
        mask = y_for_loss != -1
        y_for_loss = y_for_loss[mask]
        y_hat_for_loss = y_hat[mask]

        loss = nn.CrossEntropyLoss()(y_hat_for_loss, y_for_loss)
        self.log("eval/val_loss", loss)

        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy()

        far, _, _, _, eer_index, _ = get_eer(y, y_hat)
        eer = far[eer_index] 
        self.log("eval/eer", eer)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
