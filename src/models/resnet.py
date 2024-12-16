import torch
import torch.nn as nn
from sklearn.metrics import f1_score

import lightning as L
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights

import warnings
warnings.filterwarnings("ignore")



class Resnet(pl.LightningModule):
    def __init__(self, batch_size=32, num_classes=2000, verbose = False):
        super().__init__()
        self.batch_size = batch_size
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        else:
            raise ValueError(f"Unknown model name: {name}")
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        else:
            raise ValueError(f"Unknown model name: {name}")
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
            self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
        self.model = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unknown model name: {name}")
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


    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)


    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]



class ClosedSetClassifier():
    def __init__(self, backbone_checkpoint, batch_size=32, num_classes=2000):
        self.backbone = Resnet(batch_size, num_classes)
        self.backbone.load_from_checkpoint(backbone_checkpoint)
        self.backbone.model.fc = nn.Identity()
        self.backbone.eval()

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.model.fc.in_features, num_classes),
            nn.Softmax(dim=1),
            
        )


    def forward(self, x):
        return super().forward(x)

