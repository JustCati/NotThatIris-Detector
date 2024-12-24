import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import lightning as pl
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

import warnings
warnings.filterwarnings("ignore")



class Resnet(pl.LightningModule):
    def __init__(self, batch_size=32, num_classes=2000, verbose = False):
        super().__init__()
        self._y = []
        self._y_pred = []
        self._y_loss = []
        self._y_hat_loss = []
        self._training_loss = []
        self._loss_value = 1_000_000.0 # Just a big number

        self.batch_size = batch_size
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        if verbose:
            print(self.model)


    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.view(-1).long()
        y_hat = self(x.to(self.device))

        loss = nn.CrossEntropyLoss()(y_hat, y)
        self._training_loss.append(loss.item())
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x.to(self.device))

        y_loss = y.view(-1).long()
        self._y_loss.extend(y_loss.cpu().numpy())
        self._y_hat_loss.extend(y_hat.cpu().numpy())

        y = y.cpu().numpy()
        y_hat = y_hat.argmax(dim=1).cpu().numpy()
        self._y.extend(y)
        self._y_pred.extend(y_hat)


    def on_train_epoch_end(self):
        loss = torch.tensor(self._training_loss).mean()
        self._loss_value = loss
        self._training_loss = []


    def on_validation_epoch_end(self):
        y_loss = torch.tensor(self._y_loss)
        y_hat_loss = torch.tensor(self._y_hat_loss)

        loss = nn.CrossEntropyLoss()(y_hat_loss, y_loss)
        self.log("eval/val_loss", loss)
        self.log("train/loss", self._loss_value)

        f1 = f1_score(self._y, self._y_pred, average="macro")
        self.log("eval/f1", f1)

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



class FeatureExtractor(pl.LightningModule):
    def __init__(self, model=None, model_path=None, num_classes=819):
        super().__init__()
        if model is not None:
            self.model = model
        elif model_path is not None:
            num_classes = num_classes if num_classes is not None else 819
            self.model = Resnet.load_from_checkpoint(model_path, num_classes=num_classes)
        else:
            self.model = Resnet(num_classes=num_classes)
        self.model.model.fc = nn.Identity()
        self.model.eval()


    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
