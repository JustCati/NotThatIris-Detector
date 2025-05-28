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
                 classes,
                 transform=None,
                 p=0.5,
                 keep_unknown=False):
        self.p = p
        self.transform = transform
        self.keep_unknown = keep_unknown

        self.gt = self.__process_df(csv_file)
        self.__sanitize()
        
        self.label_map = self.__create_label_map(classes)
        self.num_classes = len(self.label_map)


    def __sanitize(self):
        toRemove = []
        for i in range(len(self.gt)):
            if not os.path.exists(self.gt.loc[i]["ImagePath"]):
                toRemove.append(i)
        self.gt = self.gt.drop(toRemove)
        self.gt = self.gt.reset_index(drop=True)
        if not self.keep_unknown:
            self.gt = self.gt.drop(self.gt[self.gt["Label"] == -1].index)


    def get_active_labels(self):
        return sorted(self.gt["Label"].unique())


    def get_mapper(self):
        return {v: k for k, v in self.label_map.items()}


    def __create_label_map(self, classes):
        label_map = {label: i for i, label in enumerate(classes)}
        label_map.update({-1: -1})
        return label_map


    def __process_df(self, csv_file):
        df = pd.read_csv(csv_file, index_col=0)
        df["ImagePath"] = df["ImagePath"].apply(
            lambda x: x.replace(
                "images",
                "normalized"
            )
        )
        return df


    def __getitem__(self, idx):
        img_path = self.gt.loc[idx, "ImagePath"]
        label = self.label_map[self.gt.loc[idx, "Label"]]

        img = Image.open(img_path)
        if self.transform:
            if torch.rand(1) < self.p:
                img = self.transform(img)

        label = torch.tensor([label])
        img = transforms.ToTensor()(img)
        return img, label


    def __len__(self):
        return len(self.gt)
