import cv2
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
        return torch.sum(weight * (input - target) ** 2) / torch.sum(weight)


def weighted_psnr_per_image(denoised, ground_truth, weight, max_pixel_value=1.0):
    """
    Computes the average weighted PSNR over a batch by calculating PSNR for each image individually.
    
    Parameters:
    - denoised: Tensor of shape (N, C, H, W)
    - ground_truth: Tensor of shape (N, C, H, W)
    - weight: Tensor of shape (N, C, H, W)
    - max_pixel_value: Maximum possible pixel value (default is 1.0 for normalized images)
    
    Returns:
    - Average PSNR over the batch
    """
    batch_size = denoised.size(0)
    psnr_values = []
    for i in range(batch_size):
        mse = torch.sum(weight[i] * (denoised[i] - ground_truth[i]) ** 2) / torch.sum(weight[i])
        psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
        psnr_values.append(psnr)
    return torch.stack(psnr_values).mean()



class DNCNN(pl.LightningModule):
    def __init__(self, verbose = False):
        super().__init__()
        self._y = []
        self._masks = []
        self._y_pred = []
        self._y_loss = []
        self._y_hat_loss = []
        self._training_loss = []
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
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y, mask = y
        y_hat = self(x.to(self.device))

        self._y_loss.extend(y_hat.cpu().numpy())
        self._y_hat_loss.extend(y_hat.cpu().numpy())

        y = y.cpu().numpy()
        y_hat = y_hat.cpu().numpy()
        self._y.extend(y)
        self._y_pred.extend(y_hat)
        self._masks.extend(mask.cpu().numpy())


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
        masks_all_np = np.concatenate(self._masks, axis=0)

        gt_tensor = torch.from_numpy(ground_truth_all_np).to(self.device)
        pred_tensor = torch.from_numpy(predictions_all_np).to(self.device)
        mask_tensor = torch.from_numpy(masks_all_np).to(self.device)

        val_loss_epoch = self.criterion(pred_tensor, gt_tensor, mask_tensor)

        self.log("eval/val_loss", val_loss_epoch)
        self.log("train/loss", self._loss_value)

        gt_for_psnr = ground_truth_all_np.astype(np.uint8)
        pred_for_psnr = predictions_all_np.astype(np.uint8)
        masks_all_np = masks_all_np.astype(np.float32)

        psnr_value = weighted_psnr_per_image(
            denoised=torch.from_numpy(pred_for_psnr).to(self.device),
            ground_truth=torch.from_numpy(gt_for_psnr).to(self.device),
            weight=torch.from_numpy(masks_all_np).to(self.device),
            max_pixel_value=255.0
        )
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
