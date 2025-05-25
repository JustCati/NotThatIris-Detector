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
        if not self._y or not self._y_pred:
            self.log("eval/val_loss", float('nan'), on_step=False, on_epoch=True, prog_bar=True)
            self.log("eval/psnr", float('nan'), on_step=False, on_epoch=True, prog_bar=True)
            self._y = []
            self._y_pred = []
            return

        ground_truth_all_np = np.concatenate(self._y, axis=0)
        predictions_all_np = np.concatenate(self._y_pred, axis=0)

        gt_tensor = torch.from_numpy(ground_truth_all_np).to(self.device)
        pred_tensor = torch.from_numpy(predictions_all_np).to(self.device)

        sum_squared_errors = self.criterion(pred_tensor, gt_tensor)        
        num_images = ground_truth_all_np.shape[0]
        val_loss_epoch = sum_squared_errors / (num_images * 2)

        self.log("eval/val_loss", val_loss_epoch)
        self.log("train/loss", self._loss_value)

        gt_for_psnr = ground_truth_all_np.astype(np.uint8)
        pred_for_psnr = predictions_all_np.astype(np.uint8)

        psnr_value = cv2.PSNR(gt_for_psnr, pred_for_psnr)
        self.log("eval/psnr", psnr_value)

        self._y = []
        self._y_pred = []



    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
