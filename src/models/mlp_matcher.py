import torch
import torch.nn as nn
import lightning as pl
from src.engine.thresholding import get_eer



class MLPMatcher(pl.LightningModule):
    def __init__(self, in_feature, num_classes, threshold=None, extractor=None, verbose=False):
        super().__init__()
        self._val_y = []
        self._val_y_pred = []
        self._losses = []
        self._loss_value = 1_000_000.0 

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
            x = torch.softmax(x, dim=1)
            mask = x > self.threshold
            x[~mask] = 0
        return x


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1).long()
        y_hat = self(x.to(self.device))

        loss = nn.CrossEntropyLoss()(y_hat, y)
        self._losses.append(loss.item())
        return loss


    def on_train_epoch_end(self):
        mean_loss = torch.tensor(self._losses).mean()
        self._loss_value = mean_loss
        self._losses = []


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))

        y_for_loss = y.view(-1).long()
        mask = y_for_loss != -1
        y_for_loss = y_for_loss[mask]

        y[mask] = 1
        y = y.cpu().numpy()
        y_hat = torch.softmax(y_hat, dim=1)
        y_hat = y_hat.max(dim=1).values.cpu().numpy()

        self._val_y.extend(y)
        self._val_y_pred.extend(y_hat)


    def on_validation_epoch_end(self):
        _, frr, _, _, eer_index, _ = get_eer(self._val_y, self._val_y_pred)
        self.log("eval/eer", frr[eer_index])
        self.log("train/loss", self._loss_value)

        self._val_y = []
        self._val_y_pred = []


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
