import os
import argparse

from src.utils.utils import get_device
from src.models.yolo import getYOLO, train as yolo_train
from src.utils.dataset_utils.yolo import convert_ann_to_seg, generate_annotations, split_data, augment_data

import warnings
warnings.filterwarnings("ignore")




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
    dataset_path = os.path.join(os.path.dirname(__file__), data_path, "Iris-Degradation")
    ann = os.path.join(dataset_path, 'masks')
    yolo_ann = os.path.join(dataset_path, 'labels')
    yaml_path = os.path.join(dataset_path, f'Iris-Degradation.yaml')

    if not os.path.exists(os.path.join(ann, "train")):
        split_data(dataset_path=dataset_path)

    # if not os.path.exists(ann):
    #     generate_annotations(dataset_path=dataset_path)

    if not os.path.exists(yolo_ann):
        convert_ann_to_seg(ann_path=ann,
                           out_path=yolo_ann,
                           classes=1)

    if not os.path.exists(os.path.join(dataset_path, 'images')):
        augment_data(dataset_path=os.path.join(dataset_path, 'images_raw'),
                      out_path=os.path.join(dataset_path, 'images'))

    #* Format yaml file
    if not os.path.exists(yaml_path):
        raise ValueError('Yaml file does not exist')

    #* Load YOLO model
    device = get_device()
    model_file = args.checkpoint if args.checkpoint != '' else 'last.pt'
    yolo_checkpoint_path = "" if scratch else os.path.join(model_path, "weights", model_file)

    yolo_model = getYOLO(checkpoint_path=yolo_checkpoint_path,
                         task='segment',
                         device=device,
                         inference=False)
    print("YOLO model loaded successfully")

    # #* Train YOLO model
    epochs = args.epochs
    batch_size = args.batch_size
    yolo_train(model=yolo_model, 
                yaml_file=yaml_path,
                batch_size=batch_size,
                epochs=epochs,
                patience=args.patience,
                model_path=model_path if scratch else os.path.dirname(model_path),
                folder_name=os.path.basename(model_path) if not scratch else "YOLOSEG",
                resume=not scratch,
                device=device)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='datasets', help='Path to data folder')
    parser.add_argument('--model_path', type=str, default='/', help='Path to model checkpoints folder')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=0, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--checkpoint', type=str, default='', help='Name of checkpoint file to resume training')
    args = parser.parse_args()
    main(args)
