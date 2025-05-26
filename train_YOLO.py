import os
import argparse

import multiprocessing
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2 as T
from concurrent.futures import ThreadPoolExecutor

from src.utils.utils import get_device
from src.models.yolo import getYOLO, train as yolo_train
from src.utils.dataset_utils.yolo import convert_ann_to_yolo

import warnings
warnings.filterwarnings("ignore")



CLASSESS = {
    "left_eye" : 5,
    "right_eye" : 6
}
CONVERT_CLASS = {
    5 : 0,
    6 : 0 # Both eyes are the same class
}


def process_file(in_path, out_path):
    pipeline = T.Compose([
        T.Grayscale(num_output_channels=3),
    ])
    
    img = Image.open(in_path)
    try:
        updated_img = pipeline(img)
    except Exception as e:
        print(f"Error processing {in_path}: {e}")
        return

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    updated_img.save(out_path)


def augment_data(dataset_path, out_path):
    files_to_process = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                files_to_process.append((os.path.join(folder_path, file), os.path.join(out_path, folder, file)))

    print(f"Found {len(files_to_process)} files to process in {dataset_path}")
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        list(tqdm(executor.map(lambda x: process_file(*x), files_to_process), total=len(files_to_process)))



def main(args):
    data_path = args.path
    if not os.path.exists(data_path):
        raise ValueError('Data path does not exist')

    scratch = False
    model_path = args.model_path
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    if model_path == '':
        scratch = True
        model_path = os.path.join(os.path.dirname(__file__), 'ckpts', "YOLO")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    #* 1. TRAIN YOLO
    #* Convert annotations to YOLO format
    portrait_dataset_path = os.path.join(os.path.dirname(__file__), data_path, args.dataset)
    portrait_ann = os.path.join(portrait_dataset_path, 'annotations')
    yolo_portrait_ann = os.path.join(portrait_dataset_path, 'labels')
    easy_portrait_yaml_path = os.path.join(portrait_dataset_path, f'{args.dataset}.yaml')

    if not os.path.exists(yolo_portrait_ann):
        os.makedirs(yolo_portrait_ann)

    src_ann_count = sum([len(files) for _, _, files in os.walk(portrait_ann)]) - 1
    dst_ann_count = sum([len(files) for _, _, files in os.walk(yolo_portrait_ann)])
    if "train.cache" in os.listdir(yolo_portrait_ann) and "test.cache" in os.listdir(yolo_portrait_ann):
        dst_ann_count -= 2
    print(f"Source annotations count: {src_ann_count}")
    print(f"Destination annotations count: {dst_ann_count}")
    if src_ann_count != dst_ann_count:
        convert_ann_to_yolo(portrait_ann, yolo_portrait_ann, CLASSESS, CONVERT_CLASS)

    if not os.path.exists(os.path.join(portrait_dataset_path, 'images')):
        augment_data(dataset_path=os.path.join(portrait_dataset_path, 'images_raw'),
                      out_path=os.path.join(portrait_dataset_path, 'images'))

    #* Format yaml file
    if not os.path.exists(easy_portrait_yaml_path):
        raise ValueError('Easy portrait yaml file does not exist')

    #* Load YOLO model
    device = get_device()
    model_file = args.checkpoint if args.checkpoint != '' else 'last.pt'
    yolo_checkpoint_path = None if scratch else os.path.join(model_path, "weights", model_file)

    yolo_model = getYOLO(checkpoint_path=yolo_checkpoint_path, task="detection", device=device, inference=False)
    print("YOLO model loaded successfully")

    # #* Train YOLO model
    epochs = args.epochs
    batch_size = args.batch_size
    yolo_train(model=yolo_model, 
                yaml_file=easy_portrait_yaml_path,
                batch_size=batch_size,
                epochs=epochs,
                patience=args.patience,
                model_path=model_path if scratch else os.path.dirname(model_path),
                folder_name=os.path.basename(model_path) if not scratch else "YOLODET",
                resume=not scratch,
                device=device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='datasets', help='Path to data folder')
    parser.add_argument('--dataset', type=str, default='EasyPortrait', help='Dataset name')
    parser.add_argument('--model_path', type=str, default='/', help='Path to model checkpoints folder')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--checkpoint', type=str, default='', help='Name of checkpoint file to resume training')
    args = parser.parse_args()
    main(args)
