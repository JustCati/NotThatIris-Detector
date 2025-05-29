import math
import torch
import numpy as np
import torch.nn as nn
import lightning as pl
from external.DnCNN.models import DnCNN


import warnings
warnings.filterwarnings("ignore")


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, input, target, weight):
        return torch.sum((weight * (input - target)) ** 2) / torch.sum(weight)


def weighted_psnr_per_image(denoised, ground_truth, weight, max_pixel_value=1.0):
    batch_size = denoised.size(0)
    psnr_values = []
    for i in range(batch_size):
        mse = torch.sum((weight[i] * (denoised[i] - ground_truth[i])) ** 2) / torch.sum(weight[i])
        psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
        psnr_values.append(psnr)
    return torch.stack(psnr_values).mean()


class DNCNN(pl.LightningModule):
    def __init__(self, verbose = False):
        super().__init__()
        self._training_loss = []
        self._batch_val_loss = []
        self._batch_val_psnrs = []
        self._loss_value = 1_000_000.0 # Just a big number


        self.model = DnCNN(channels=3, num_of_layers=20)
        self.model = self.model.apply(self.__weights_init_kaiming)
        self.criterion = WeightedMSELoss()

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
        y, mask = y
        y_hat = self(x.to(self.device))

        loss = self.criterion(y_hat, y.to(self.device), mask.to(self.device))
        self._training_loss.append(loss.item())
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y_data = batch
        y_true, mask = y_data
        y_pred = self(x.to(self.device))

        y_true = y_true.to(self.device)
        mask_dev = mask.to(self.device).float()

        pred_for_psnr_batch = (y_pred.detach() * 255.0).clamp(0, 255).byte()
        gt_for_psnr_batch = (y_true * 255.0).clamp(0, 255).byte()

        batch_psnr = weighted_psnr_per_image(
            denoised=pred_for_psnr_batch,
            ground_truth=gt_for_psnr_batch,
            weight=mask_dev,
            max_pixel_value=255.0
        )
        self._batch_val_psnrs.append(batch_psnr.item())
        
        batch_loss = self.criterion(y_pred, y_true, mask_dev)
        self._batch_val_loss.append(batch_loss.item())


    def on_train_epoch_end(self):
        loss = torch.tensor(self._training_loss).mean()
        self._loss_value = loss
        self._training_loss = []


    def on_validation_epoch_end(self):
        epoch_loss = torch.tensor(self._batch_val_loss).mean()
        self.log("eval/val_loss", epoch_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self._loss_value)

        epoch_psnr = torch.tensor(self._batch_val_psnrs).mean()
        self.log("eval/psnr", epoch_psnr, on_step=False, on_epoch=True, prog_bar=True)

        self._batch_val_loss = []
        self._batch_val_psnrs = []


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
