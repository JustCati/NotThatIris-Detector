import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms.v2 import GaussianBlur

from basicsr.utils import img2tensor
from basicsr.data.transforms import augment
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.degradations import add_jpg_compression




@DATASET_REGISTRY.register()
class UpsampleDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_path = opt["root_path"]
        csv_file = opt["csv_file"]
        if csv_file.startswith("/"):
            csv_file = csv_file[1:]
        csv_file = os.path.join(root_path, csv_file)
        self.gt = pd.read_csv(csv_file, index_col=0)
        self.gt = self.gt["ImagePath"].apply(
            lambda x: x.replace(
                "images",
                "normalized"
            )
        ).tolist()
        self.__sanitize()


    def __sanitize(self):
        for i in range(len(self.gt)):
            if not os.path.exists(self.gt[i]):
                print(f"Image {self.gt[i]} does not exist, removing from dataset.")
                self.gt[i] = None
        self.gt = [x for x in self.gt if x is not None]


    def __getitem__(self, idx):
        img_path = self.gt[idx]
        img_gt = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img_gt = cv2.resize(img_gt, (128, 128), interpolation=cv2.INTER_LINEAR)
        img_lq = img_gt.copy()
        h, w = img_lq.shape

        scale = self.opt.get("scale", 1)
        if scale > 1:
            img_lq = cv2.resize(img_lq, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)

        img_lq = cv2.GaussianBlur(img_lq, (11, 11), 5)
        _, encoded_img = cv2.imencode('.jpg', img_lq, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        img_lq = cv2.imdecode(encoded_img, cv2.IMREAD_GRAYSCALE)
        
        img_gt = img_gt[..., np.newaxis]
        img_lq = img_lq[..., np.newaxis]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        return {"gt": img_gt, "lq": img_lq}

    def __len__(self):
        return len(self.gt)


    def __len__(self):
        return len(self.gt)
