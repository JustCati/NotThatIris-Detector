import os
import argparse

from src.utils.utils import get_device
from src.models.yolo import getYOLO, train as yolo_train
from utils.dataset_utils.yolo import convert_ann_to_yolo, read_yaml

import warnings
warnings.filterwarnings("ignore")


def main(args):
    #? Parse arguments and check if paths exist
    data_path = args.path
    if not os.path.exists(data_path):
        raise ValueError('Data path does not exist')

    scratch = False
    model_path = args.model_path
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    if model_path == '':
        scratch = True
        model_path = os.path.join(os.path.dirname(__file__), 'ckpt', "YOLO")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #* 1. TRAIN YOLO
    #* Convert annotations to YOLO format
    portrait_dataset_path = os.path.join(os.path.dirname(__file__), data_path, 'EasyPortrait')
    portrait_train = os.path.join('images', "train")
    portrait_val = os.path.join('images', "test")
    portrait_ann = os.path.join(portrait_dataset_path, 'annotations')
    yolo_portrait_ann = os.path.join(portrait_dataset_path, 'labels')
    easy_portrait_yaml_path = os.path.join(portrait_dataset_path, 'easy_portrait.yaml')

    #* Format yaml file
    if not os.path.exists(easy_portrait_yaml_path):
        raise ValueError('Easy portrait yaml file does not exist')
    easy_portrait_yaml = read_yaml(easy_portrait_yaml_path, portrait_dataset_path, portrait_train, portrait_val)

    if not os.path.exists(yolo_portrait_ann):
        os.makedirs(yolo_portrait_ann)

    src_ann_count = sum([len(files) for _, _, files in os.walk(portrait_ann)])
    dst_ann_count = sum([len(files) for _, _, files in os.walk(yolo_portrait_ann)])
    if "train.cache" in os.listdir(yolo_portrait_ann) and "test.cache" in os.listdir(yolo_portrait_ann):
        dst_ann_count -= 2
    if src_ann_count != dst_ann_count:
        convert_ann_to_yolo(portrait_ann, yolo_portrait_ann)

    #* Load YOLO model
    device = get_device()
    yolo_checkpoint_path = os.path.join("models", "yolov10l.pt") if scratch else os.path.join(model_path, "weights", "last.pt")

    yolo_model = getYOLO(checkpoint_path=yolo_checkpoint_path, device=device)
    print("YOLO model loaded successfully")

    # #* Train YOLO model
    epochs = args.epochs
    batch_size = args.batch_size
    yolo_train(model=yolo_model, 
                yaml_file=easy_portrait_yaml,
                batch_size=batch_size,
                epochs=epochs,
                model_path=model_path if scratch else os.path.dirname(model_path),
                resume=not scratch,
                device=device)






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='Path to data folder')
    parser.add_argument('--model_path', type=str, default='', help='Path to model checkpoints folder')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    args = parser.parse_args()
    main(args)
