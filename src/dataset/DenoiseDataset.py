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
        self.gt["ImagePath"] = self.gt["ImagePath"].apply(
            lambda x: x.replace(
                "images",
                "normalized"
            )
        )
        self.gt = self.gt["ImagePath"].tolist()
        self.__sanitize()


    def __sanitize(self):
        for i in range(len(self.gt)):
            if not os.path.exists(self.gt[i]):
                print(f"Image {self.gt[i]} does not exist, removing from dataset.")
                self.gt[i] = None
        self.gt = [x for x in self.gt if x is not None]


    def __getitem__(self, idx):
        img_path = self.gt[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_x = self.transform(img)

        img = transforms.ToTensor()(img)
        img_x = transforms.ToTensor()(img_x)
        return img_x, img


    def __len__(self):
        return len(self.gt)
