import os
import argparse

from src.models.yolo import getYOLO
from src.utils.utils import get_device
from src.utils.dataset_utils import convert_ann_to_yolo
from src.engines.YOLO.train import train as yolo_train

import warnings
warnings.filterwarnings("ignore")


def main(args):
    #? Parse arguments and check if paths exist
    data_path = args.path
    if not os.path.exists(data_path):
        raise ValueError('Data path does not exist')
    model_path = args.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    yolo_model_path = os.path.join(model_path, 'YOLO')
    if not os.path.exists(yolo_model_path):
        os.makedirs(yolo_model_path)

    #* 1. TRAIN YOLO
    #* Convert annotations to YOLO format
    portarait_dataset_path = os.path.join(data_path, 'EasyPortrait')
    portrait_ann = os.path.join(portarait_dataset_path, 'annotations')
    yolo_portrait_ann = os.path.join(portarait_dataset_path, 'yolo_annotations')

    if not os.path.exists(yolo_portrait_ann):
        os.makedirs(yolo_portrait_ann)

    src_ann_count = sum([len(files) for _, _, files in os.walk(portrait_ann)])
    dst_ann_count = sum([len(files) for _, _, files in os.walk(yolo_portrait_ann)])
    if src_ann_count != dst_ann_count:
        convert_ann_to_yolo(portrait_ann, yolo_portrait_ann, folder=True)

    #* Load YOLO model
    pull_from_scratch = ("model.pt" not in os.listdir(yolo_model_path))
    yolo_checkpoint_path = os.path.join("models", "yolov10l.pt") if pull_from_scratch else os.path.join(yolo_model_path, 'model.pt')
    device = get_device()

    yolo_model = getYOLO(checkpoint_path=yolo_checkpoint_path, device=device)
    print("YOLO model loaded successfully")







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='Path to data folder')
    parser.add_argument('--model_path', type=str, default='ckpts', help='Path to model checkpoints folder')
    args = parser.parse_args()
    main(args)
