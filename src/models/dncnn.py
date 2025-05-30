import math
import torch
import torch.nn as nn
import lightning as pl
from external.DnCNN.models import DnCNN


import warnings
warnings.filterwarnings("ignore")


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, input, target):
        mask = target != 0
        return torch.abs(input - target)[mask].mean()


def weighted_psnr_per_image(denoised, ground_truth, max_pixel_value=1.0):
    psnr_values = []
    batch_size = denoised.size(0)

    for i in range(batch_size):
        mask = ground_truth[i] != 0
        mse = torch.sum((denoised[i] - ground_truth[i])[mask] ** 2) / mask.sum()
        psnr = 10 * torch.log10(max_pixel_value ** 2 / mse)
        psnr_values.append(psnr)
    return torch.stack(psnr_values).mean()


class DNCNN(pl.LightningModule):
    def __init__(self, feat_extractor=None, verbose = False):
        super().__init__()
        self._training_loss = []
        self._batch_val_im_loss = []
        self._batch_val_ctx_loss = []
        self._batch_val_psnrs = []
        self._loss_value = 1_000_000.0 # Just a big number

        self.feat_extractor = feat_extractor
        self.feat_extractor.eval()
        
        self.model = DnCNN(channels=3, num_of_layers=20)
        self.model = self.model.apply(self.__weights_init_kaiming)
        
        self.context_criterion = nn.L1Loss()
        self.image_criterion = WeightedL1Loss()

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

        with torch.no_grad():
            features_hat = self.feat_extractor(y_hat)
        feature_true = self.feat_extractor(y.to(self.device))

        im_loss = self.image_criterion(y_hat, y.to(self.device))
        context_loss = self.context_criterion(features_hat, feature_true)

        loss = im_loss + context_loss
        self._training_loss.append(loss.item())
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x.to(self.device))

        y_true = y_true.to(self.device)

        pred_for_psnr_batch = (y_pred.detach() * 255.0).clamp(0, 255).byte()
        gt_for_psnr_batch = (y_true * 255.0).clamp(0, 255).byte()

        batch_psnr = weighted_psnr_per_image(
            denoised=pred_for_psnr_batch,
            ground_truth=gt_for_psnr_batch,
            max_pixel_value=255.0
        )
        self._batch_val_psnrs.append(batch_psnr.item())
        
        batch_im_loss = self.image_criterion(y_pred, y_true)
        batch_ctx_loss = self.context_criterion(
            self.feat_extractor(y_pred),
            self.feat_extractor(y_true)
        )
        
        self._batch_val_im_loss.append(batch_im_loss.item())
        self._batch_val_ctx_loss.append(batch_ctx_loss.item())


    def on_train_epoch_end(self):
        loss = torch.tensor(self._training_loss).mean()
        self._loss_value = loss
        self._training_loss = []


    def on_validation_epoch_end(self):
        epoch_loss = torch.tensor(self._batch_val_im_loss).mean()
        self.log("eval/val_im_loss", epoch_loss, on_step=False, on_epoch=True, prog_bar=True)
        epoch_ctx_loss = torch.tensor(self._batch_val_ctx_loss).mean()
        self.log("eval/val_ctx_loss", epoch_ctx_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self._loss_value)

        epoch_psnr = torch.tensor(self._batch_val_psnrs).mean()
        self.log("eval/psnr", epoch_psnr, on_step=False, on_epoch=True, prog_bar=True)

        final_eval = epoch_psnr - epoch_ctx_loss
        self.log("eval/final_eval", final_eval, on_step=False, on_epoch=True, prog_bar=True)

        self._batch_val_im_loss = []
        self._batch_val_psnrs = []
        self._batch_val_ctx_loss = []


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
