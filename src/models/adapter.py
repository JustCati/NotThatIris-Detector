import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F




class Adapter(pl.LightningModule):
    def __init__(self, in_features, margin=1, verbose=False):
        super().__init__()
        self._anchors = []
        self._positives = []
        self._negatives = []
        self._losses = []
        self._loss_value = 1_000_000.0 # Just a high value

        self.margin = margin
        self.criterion = nn.TripletMarginLoss(margin=margin)
        self.Adapter = nn.Linear(in_features, in_features)
        if verbose:
            print(self.Adapter)


    def forward(self, x):
        return self.Adapter(x)


    def training_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

        loss = self.criterion(anchor, positive, negative)
        self._losses.append(loss.item())
        return loss


    def on_train_epoch_end(self):
        loss = torch.tensor(self._losses).mean()
        self._loss_value = loss.item()
        self._losses = []


    def validation_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

        self._anchors.extend(anchor.cpu().numpy())
        self._positives.extend(positive.cpu().numpy())
        self._negatives.extend(negative.cpu().numpy())


    def _triplet_accuracy(self, anchor, positive, negative):
        accuracy = (F.pairwise_distance(anchor, positive) < F.pairwise_distance(anchor, negative)).sum().item() / len(anchor)
        return accuracy


    def on_validation_epoch_end(self):
        anchor = torch.tensor(self._anchors)
        positive = torch.tensor(self._positives)
        negative = torch.tensor(self._negatives)

        loss = self.criterion(anchor, positive, negative)
        self.log("train/train_loss", self._loss_value)
        self.log("eval/val_loss", loss)

        accuracy = self._triplet_accuracy(anchor, positive, negative)
        self.log("eval/accuracy", accuracy)

        self._anchors = []
        self._positives = []
        self._negatives = []


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]




class FeatureAdapter(pl.LightningModule):
    def __init__(self, model=None, model_path=None, num_classes=2048):
        super().__init__()
        if model is not None:
            self.model = model
        elif model_path is not None:
            num_classes = num_classes if num_classes is not None else 2048
            self.model = Adapter.load_from_checkpoint(model_path, in_features=num_classes)
        else:
            self.model = Adapter(in_features=2048)
        self.model.eval()


    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
