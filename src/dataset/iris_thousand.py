import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset



class IrisThousand(Dataset):
    def __init__(self, csv_file, dataset_path, transform=None):
        self.transform = transform
        self.gt, self.label_map = self.__process_df(csv_file, dataset_path)


    def __process_df(self, csv_file, dataset_path):
        df = pd.read_csv(csv_file, index_col=0)
        df["ImagePath"] = df["ImagePath"].apply(
            lambda x: x.replace(
                "/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand",
                dataset_path
            )
        )
        df = df.reset_index(drop=True)
        label_map = {label: i for i, label in enumerate(sorted(df["Label"].unique()))}
        return df, label_map


    def __getitem__(self, idx):
        img_path = self.gt.loc[idx, "ImagePath"]
        label = self.label_map[self.gt.loc[idx, "Label"]]

        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)

        label = torch.tensor([label])
        img = transforms.ToTensor()(img)
        return img, label


    def __len__(self):
        return len(self.gt)