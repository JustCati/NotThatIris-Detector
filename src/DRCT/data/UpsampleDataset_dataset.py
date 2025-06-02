import os
import cv2
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
        
        hq = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if hq.size[0] != 128:
            hq = cv2.resize(hq, (128, 128), interpolation=cv2.INTER_LINEAR)
        lq = hq.copy()

        h, w = hq.shape[:2]
        scale = self.opt["scale"]
        lq = cv2.resize(lq, (w // scale, h // scale), interpolation=cv2.INTER_LINEAR)

        lq = cv2.transpose(lq, (2, 0, 1))
        lq = GaussianBlur(kernel_size=11, sigma=5)(lq)
        lq = cv2.transpose(lq, (1, 2, 0))
        lq = add_jpg_compression(lq, quality=50)
        
        if self.opt['phase'] == 'train':
            img_gt, img_lq = augment([hq, lq], self.opt['use_hflip'], self.opt['use_rot'])

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        return {"gt": img_gt, "lq": img_lq}


    def __len__(self):
        return len(self.gt)
