import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class IrisThousand(Dataset):
    def __init__(self, csv_file, dataset_path, original_csv_file, transform=None, p=0.5):
        self.p = p
        self.transform = transform
        self.gt = self.__process_df(csv_file, dataset_path)
        self.label_map = self.__create_label_map(original_csv_file)
        self.num_classes = len(self.label_map)


    def __create_label_map(self, original_csv_file):
        df = pd.read_csv(original_csv_file, index_col=0)
        label_map = {label: i for i, label in enumerate(df["Label"].unique())}
        label_map.update({"-1": -1})
        return label_map


    def __process_df(self, csv_file, dataset_path):
        df = pd.read_csv(csv_file, index_col=0)
        df["ImagePath"] = df["ImagePath"].apply(
            lambda x: x.replace(
                "/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand",
                dataset_path
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
