import torch
import numpy as np
import torch.nn as nn
import lightning as pl
from src.models.backbone import FeatureExtractor
from src.models.esrgan.ESRGAN import Generator, Discriminator


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





class GAN(pl.LightningModule):
    def __init__(self, image_shape, feat_extractor=None, verbose = False):
        super().__init__()
        self.automatic_optimization = False
        
        self._training_gen_loss = []
        self._training_dis_loss = []
        self._batch_val_psnrs = []
        self._batch_val_im_loss = []
        self._batch_val_ctx_loss = []
        self._loss_value = 1_000_000.0 # Just a big number

        self.image_shape = image_shape

        if feat_extractor is not None:
            self.feat_extractor = feat_extractor
            self.feat_extractor.eval()
        else:
            self.feat_extractor = FeatureExtractor()
        
        self.generator = Generator(channels=3, num_upsample=1, num_res_blocks=23, factor=1)
        self.discriminator = Discriminator(input_shape=(3, *self.image_shape))
        
        self.context_criterion = nn.L1Loss()
        self.image_criterion = WeightedL1Loss()
        self.discriminator_criterion = nn.BCEWithLogitsLoss()

        if verbose:
            print(self.generator)
            print(self.discriminator)


    def forward(self, x):
        return self.generator(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        optimizer_g, optimizer_d = self.optimizers()
        schuduler_g, scheduler_d = self.lr_schedulers()

        self.toggle_optimizer(optimizer_g)
        pred = self(x.to(self.device))
        gen_loss = self.image_criterion(pred, y.to(self.device))
        
        pred_fake = self.discriminator(pred.detach())
        valid = torch.ones(x.size(0), *self.discriminator.output_shape).type_as(x).to(self.device)
        dis_loss = self.discriminator_criterion(pred_fake, valid)
        
        with torch.no_grad():
            features_hat = self.feat_extractor(pred)
        feature_true = self.feat_extractor(y.to(self.device))
        context_loss = self.context_criterion(features_hat, feature_true)

        generator_loss = context_loss + gen_loss * 0.5 + dis_loss * 0.5
        self.log("train/generator_loss", gen_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/discriminator_loss", dis_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/context_loss", context_loss, on_step=True, on_epoch=False, prog_bar=True)

        self.manual_backward(generator_loss)
        optimizer_g.step()
        schuduler_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)
        

        self.toggle_optimizer(optimizer_d)
        valid = torch.ones(x.size(0), *self.discriminator.output_shape).type_as(x).to(self.device)
        pred_real = self.discriminator(y.to(self.device))
        real_loss = self.discriminator_criterion(pred_real, valid)

        fake = torch.zeros(x.size(0), *self.discriminator.output_shape).type_as(x).to(self.device)
        pred_fake = self.discriminator(pred.detach())
        fake_loss = self.discriminator_criterion(pred_fake, fake)

        d_loss = (real_loss + fake_loss) * 0.5
        self.log("train/discriminator_loss", d_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.manual_backward(d_loss)
        optimizer_d.step()
        scheduler_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        self._training_gen_loss.append(gen_loss.item())
        self._training_dis_loss.append(d_loss.item())
        return None        


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
        gen_loss = torch.tensor(self._training_gen_loss).mean()
        self._loss_value = gen_loss
        self._training_loss = []
        self._training_dis_loss = []


    def on_validation_epoch_end(self):
        epoch_gen_loss = torch.tensor(self._batch_val_im_loss).mean()
        self.log("eval/val_im_loss", epoch_gen_loss, on_step=False, on_epoch=True, prog_bar=True)
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
        gen_parameters = [p for p in self.parameters() if p.requires_grad]
        dis_parameters = [p for p in self.discriminator.parameters() if p.requires_grad]
        
        dis_optimizer = torch.optim.Adam(dis_parameters, lr=1e-3)
        gen_optimizer = torch.optim.Adam(gen_parameters, lr=1e-3)
        
        dis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dis_optimizer, T_max=10)
        gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=10)
        return [dis_optimizer, gen_optimizer], [dis_scheduler, gen_scheduler]
