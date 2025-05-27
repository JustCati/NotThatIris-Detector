import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class NormalizedIrisDataset(Dataset):
    def __init__(self, 
                 csv_file,
                 transform=None,):
        self.transform = transform
        self.gt = pd.read_csv(csv_file, index_col=0)
        self.gt = self.gt["ImagePath"].apply(
            lambda x: x.replace(
                "images",
                "normalized"
            )
        ).tolist()
        self.masks = [elem.replace("normalized", "masks") for elem in self.gt]
        self.__sanitize()


    def __sanitize(self):
        for i in range(len(self.gt)):
            if not os.path.exists(self.gt[i]):
                print(f"Image {self.gt[i]} does not exist, removing from dataset.")
                self.gt[i] = None
                self.masks[i] = None
        self.gt = [x for x in self.gt if x is not None]
        self.masks = [x for x in self.masks if x is not None]


    def __getitem__(self, idx):
        img_path = self.gt[idx]
        mask_path = self.masks[idx]
        
        img_y = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        img_x = img_y.copy()

        if self.transform:
            img_x = self.transform(img_x.copy())

        img_x_tensor = transforms.ToTensor()(img_x)
        img_y_tensor = transforms.ToTensor()(img_y)
        mask_tensor = transforms.ToTensor()(mask)
        return img_x_tensor, (img_y_tensor, mask_tensor)


    def __len__(self):
        return len(self.gt)
