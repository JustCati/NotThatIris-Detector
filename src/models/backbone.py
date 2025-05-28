import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import lightning as pl
from torchvision.models import efficientnet_v2_s

import warnings
warnings.filterwarnings("ignore")



class EfficientNet(pl.LightningModule):
    def __init__(self, num_classes=2000, verbose = False):
        super().__init__()
        self._y = []
        self._y_pred = []
        self._y_loss = []
        self._y_hat_loss = []
        self._training_loss = []
        self._loss_value = 1_000_000.0 # Just a big number

        self.model = efficientnet_v2_s(weights="DEFAULT")
        self.vector_dim = self.model.classifier[1].in_features
        
        if num_classes is not None:
            self.model.classifier = nn.Linear(self.vector_dim, num_classes)
        
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
    def __init__(self, model=None, model_path=None):
        super().__init__()
        if model is not None:
            self.model = model
        elif model_path is not None:
            state_dict = torch.load(model_path, map_location="cpu")["state_dict"]
            src_num_classes = state_dict["model.classifier.weight"].shape[0]
            self.model = EfficientNet.load_from_checkpoint(model_path, num_classes=src_num_classes)
        else:
            self.model = EfficientNet()
        self.model.model.classifier = nn.Identity()
        self.model.eval()


    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    
    def get_vector_dim(self):
        return self.model.vector_dim
