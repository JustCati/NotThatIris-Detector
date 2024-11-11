import argparse
import os.path as osp

from src.utils.dataset_utils.drct import create_dataset

from external.DRCT.drct.archs import *
from external.DRCT.drct.data import *
from external.DRCT.drct.models import *
from basicsr.train import train_pipeline

import warnings
warnings.filterwarnings("ignore")



def main(args):
    data_path = args.path
    if not osp.exists(data_path):
        raise FileNotFoundError(f'{data_path} does not exist')

    iris_path = osp.join(data_path, 'Iris-Thousand')
    images_path = osp.join(iris_path, 'images')
    dataset_path = osp.join(iris_path, 'data')
    ckpt_path = osp.join('ckpts', 'DRCT')

    TRAIN_SPLIT_RATIO = 0.8
    SCALE_FACTOR = args.down_scale_factor
    create_dataset(images_path, dataset_path, SCALE_FACTOR, TRAIN_SPLIT_RATIO)

    #* Train DRCT model
    train_pipeline(ckpt_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='datasets', help='directory of the data')
    parser.add_argument("--train", type=bool, default=True, help='train or test')
    parser.add_argument("-opt", type=str, default='', help='path to the option yaml file')
    parser.add_argument('--force_yml', nargs='+', default=None, help='Force to update yml files. Examples: train:ema_decay=0.999')
    parser.add_argument('--down_scale_factor', type=int, default=4, help='down scale factor for generating low quality images')
    args = parser.parse_args()
    main(args)
