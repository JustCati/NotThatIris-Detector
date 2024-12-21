import torch
import lightning as pl




class GenericFeatureExtractor(pl.LightningModule):
    def __init__(self, modules: list = []):
        super().__init__()

        if len(modules) == 0:
            raise ValueError("No modules provided")
        for elem in modules:
            if not isinstance(elem, torch.nn.Module) and not isinstance(elem, pl.LightningModule):
                raise ValueError(f"Module {elem} is not a torch module")

        self.components = modules
        self.num_modules = len(modules)


    def forward(self, x):
        with torch.no_grad():
            for module in self.components:
                x = module(x)
            return x
