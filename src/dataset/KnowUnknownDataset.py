import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class KnowUnknownDataset(Dataset):
    def __init__(self, 
                 csv_file,
                 dataset_path,
                 features_only=False,
                 upsample=False,
                 transform=None,
                 p=0.5):
        self.p = p
        self.transform = transform
        self.use_upsampled = upsample
        self.features_only = features_only

        self.known_df, self.unknown_df = self.__process_df(csv_file, dataset_path)


    def get_num_classes(self):
        return len(self.known_df["Label"].unique())


    def __process_df(self, csv_file, dataset_path):
        df = pd.read_csv(csv_file, index_col=0)
        df["ImagePath"] = df["ImagePath"].apply(
            lambda x: x.replace(
                "/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand",
                dataset_path
            )
        )
        if self.use_upsampled:
            df["ImagePath"] = df["ImagePath"].apply(
                lambda x: x.replace("normalized_iris", "upsampled_iris")
            )
        if self.features_only:
            df["ImagePath"] = df["ImagePath"].apply(
                lambda x: x.replace("normalized_iris", "feature_iris").replace(".jpg", ".npy")
            )

        known_df = df[df["Label"] != "-1"]
        unknown_df = df[df["Label"] == "-1"]
        return known_df, unknown_df


    def _get_imgs(self, idx):
        anchor_img_path = self.known_df.loc[idx, "ImagePath"]
        temp_df = self.known_df.drop(idx)

        positive_img_path = temp_df.sample(1).iloc[0]["ImagePath"]
        neg_img_path = self.unknown_df.sample(1).iloc[0]["ImagePath"]

        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        neg_img = Image.open(neg_img_path)

        if self.transform:
            if torch.rand(1) < self.p:
                anchor_img = self.transform(anchor_img)

        anchor_img = transforms.ToTensor()(anchor_img)
        positive_img = transforms.ToTensor()(positive_img)
        neg_img = transforms.ToTensor()(neg_img)

        return anchor_img, positive_img, neg_img


    def _get_features(self, idx):
        anchor_path = self.known_df.loc[idx, "ImagePath"]
        temp_df = self.known_df.drop(idx)

        positive_path = temp_df.sample(1).iloc[0]["ImagePath"]
        neg_path = self.unknown_df.sample(1).iloc[0]["ImagePath"]

        anchor_feature = np.load(anchor_path)
        positive_feature = np.load(positive_path)
        neg_feature = np.load(neg_path)

        return anchor_feature, positive_feature, neg_feature


    def __getitem__(self, idx):
        if self.features_only:
            return self._get_features(idx)
        else:
            return self._get_imgs(idx)


    def __len__(self):
        return len(self.known_df)
