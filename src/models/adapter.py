import torch
import torch.nn as nn
import lightning as pl
import torch.nn.functional as F




class Adapter(pl.LightningModule):
    def __init__(self, in_features, margin=1, verbose=False):
        super().__init__()
        self.anchor = []
        self.positive = []
        self.negative = []
        self.margin = 1
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.CosineSimilarity(dim=1), margin=margin)
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

        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        loss = self.criterion(anchor, positive, negative)
        self.log("train/train_loss", loss)
        return loss


    def validation_step(self, batch, batch_idx):
        x, positive, negative = batch
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor = self(x.to(self.device))

        self.anchor.extend(anchor.cpu().numpy())
        self.positive.extend(positive.cpu().numpy())
        self.negative.extend(negative.cpu().numpy())


    def on_validation_epoch_end(self):
        anchor = torch.tensor(self.anchor)
        positive = torch.tensor(self.positive)
        negative = torch.tensor(self.negative)

        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        loss = self.criterion(anchor, positive, negative)
        self.log("eval/val_loss", loss)

        self.anchor = []
        self.positive = []
        self.negative = []


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
