import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class GenericIrisDataset(Dataset):
    def __init__(self, 
                 csv_file,
                 dataset_path,
                 original_csv_file,
                 label_map=None,
                 keep_uknown=False,
                 upsample=False,
                 modality="sample",
                 transform=None,
                 p=0.5):
        self.p = p
        self.transform = transform
        self.modality = modality == "sample"
        self.use_upsampled = upsample
        self.keep_uknown = keep_uknown

        self.gt = self.__process_df(csv_file, dataset_path)
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = self.__create_label_map(original_csv_file)
        self.num_classes = len(self.label_map)


    def get_active_labels(self):
        return sorted(self.gt["Label"].unique())


    def get_mapper(self):
        return {v: k for k, v in self.label_map.items()}


    def __create_label_map(self, original_csv_file):
        df = pd.read_csv(original_csv_file, index_col=0)
        label_map = {label: i for i, label in enumerate(df["Label"].unique())}
        label_map.update({"-1": -1}) #? Useful in case of MLP with unknown classes for open-set recognition
        return label_map


    def __process_df(self, csv_file, dataset_path):
        df = pd.read_csv(csv_file, index_col=0)
        df["ImagePath"] = df["ImagePath"].apply(
            lambda x: x.replace(
                "/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand",
                dataset_path
            )
        )
        if "608-R" in df["Label"].unique():
            df = df.drop(df[df["Label"] == "608-R"].index)
            df = df.reset_index(drop=True)
        if "747-L" in df["Label"].unique():
            df = df.drop(df[df["Label"] == "747-L"].index)
            df = df.reset_index(drop=True)
        if self.use_upsampled:
            df["ImagePath"] = df["ImagePath"].apply(
                lambda x: x.replace("normalized_iris", "upsampled_iris")
            )
        if not self.keep_uknown:
            df = df.drop(df[df["Label"] == "-1"].index)
        return df


    def __return_sample(self, idx):
        img_path = self.gt.loc[idx, "ImagePath"]
        label = self.label_map[self.gt.loc[idx, "Label"]]

        img = Image.open(img_path)
        if self.transform:
            if torch.rand(1) < self.p:
                img = self.transform(img)

        label = torch.tensor([label])
        img = transforms.ToTensor()(img)
        return img, label


    def __return_user_imgs(self, idx):
        user = self.gt.loc[idx, "Label"]
        label = self.label_map[user]

        imgs = []
        for i in range(len(self.gt[self.gt["Label"] == user])):
            img_path = self.gt[self.gt["Label"] == user].iloc[i]["ImagePath"]
            img = Image.open(img_path)

            if self.transform:
                if torch.rand(1) < self.p:
                    img = self.transform(img)
            imgs.append(transforms.ToTensor()(img))

        imgs = torch.stack(imgs)
        label = torch.tensor([label])
        return imgs, label


    def __getitem__(self, idx):
        return self.__return_sample(idx) if self.modality else self.__return_user_imgs(idx)


    def __len__(self):
        if self.modality:
            return len(self.gt)
        else:
            return len(self.gt["Label"].unique())
