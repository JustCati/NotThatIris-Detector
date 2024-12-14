import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import lightning as L
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

import warnings
warnings.filterwarnings("ignore")



class Resnet(L.LightningModule):
    def __init__(self, batch_size=32, num_classes=2000):
        super().__init__()
        self.batch_size = batch_size
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        print(self.model)


    def forward(self, x):
        return self.model(x)


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
        loss = nn.CrossEntropyLoss()(y_hat, y_for_loss)
        self.log("eval/val_loss", loss)

        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy()
        f1 = f1_score(y, y_hat, average="macro")
        self.log("eval/f1", f1)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
