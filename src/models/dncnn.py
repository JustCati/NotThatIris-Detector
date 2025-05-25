import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import lightning as pl
from external.DnCNN.models import DnCNN


import warnings
warnings.filterwarnings("ignore")



class DNCNN(pl.LightningModule):
    def __init__(self, verbose = False):
        super().__init__()
        self._y = []
        self._y_pred = []
        self._y_loss = []
        self._y_hat_loss = []
        self._training_loss = []
        self._loss_value = 1_000_000.0 # Just a big number

        self.model = DnCNN(channels=3, num_of_layers=20)
        self.model = self.model.apply(self.__weights_init_kaiming)
        self.criterion = nn.MSELoss(size_average=False)

        if verbose:
            print(self.model)


    def __weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant(m.bias.data, 0.0)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))

        loss = self.criterion(y_hat, y.to(self.device)) / (x.size()[0]*2)
        self._training_loss.append(loss.item())
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))

        self._y_loss.extend(y_hat.cpu().numpy())
        self._y_hat_loss.extend(y_hat.cpu().numpy())

        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        self._y.extend(y)
        self._y_pred.extend(y_hat)


    def on_train_epoch_end(self):
        loss = torch.tensor(self._training_loss).mean()
        self._loss_value = loss
        self._training_loss = []


    def on_validation_epoch_end(self):
        y_loss = torch.tensor(self._y_loss)
        y_hat_loss = torch.tensor(self._y_hat_loss)

        loss = self.criterion(y_loss, y_hat_loss) / (y_loss[0].size()[0]*2)
        self.log("eval/val_loss", loss)
        self.log("train/loss", self._loss_value)

        psnr = cv2.PSNR(
            y_loss.numpy().astype(np.uint8),
            y_hat_loss.numpy().astype(np.uint8),
            data_range=255.0
        )
        self.log("eval/psnr", psnr)

        self._y = []
        self._y_pred = []
        self._y_loss = []
        self._y_hat_loss = []


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
