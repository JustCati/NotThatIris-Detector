import torch
import copy
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from src.utils.eyes import get_irismask
from src.utils.dataset_utils.iris import find_eyes



class IrisDetector(nn.Module):
    def __init__(self, modules: dict):
        super(IrisDetector, self).__init__()
        
        for name, module in modules.items():
            if isinstance(module, nn.Module) and module is not None:
                setattr(self, name, module)
            else:
                print(f"Warning: {name} is not a valid nn.Module or is None, skipping initialization.")
                pass


    def forward(self, x):
        data_dict = {}
        with torch.no_grad():
            for name, module in self.named_modules():
                if name == 'yolo_det':
                    x = find_eyes(module, x)
                    data_dict["det"] = copy.deepcopy(x)
                if name == 'yolo_seg' and isinstance(x, list):
                    x = [get_irismask(eye, module) for eye in x]
                    data_dict["mask"] = copy.deepcopy(x)
                    data_dict["mask"] = [Image.fromarray(mask) if isinstance(mask, np.ndarray) else mask for mask in data_dict["mask"]]
                if name == 'sr':
                    pass # Placeholder for super-resolution model if needed
                if name == 'backbone':
                    if not all(isinstance(eye, torch.Tensor) for eye in x):
                        x = [T.ToTensor()(Image.fromarray(eye)) if isinstance(eye, np.ndarray) else eye for eye in x]
                        x = [elem.to(module.device).unsqueeze(0) for elem in x]
                    x = [module(eye) for eye in x]
                    x = [elem.squeeze(0) if isinstance(elem, torch.Tensor) else elem for elem in x]
                if name == 'mlp':
                    if isinstance(x, list) and all(isinstance(item, torch.Tensor) for item in x):
                        x = torch.stack(x, dim=0)
                    x = module(x)
                    scores = x.max(dim=1).values.cpu().numpy().tolist()
                    preds = x.argmax(dim=1).cpu().numpy().tolist()
            return scores, preds, data_dict
