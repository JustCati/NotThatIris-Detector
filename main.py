import os
import argparse

from datasets.utils import convert_ann_to_yolo


def main(args):
    data_path = args.path
    if not os.path.exists(data_path):
        raise ValueError('Data path does not exist')

    portarait_dataset_path = os.path.join(data_path, 'EasyPortrait')
    portrait_ann = os.path.join(portarait_dataset_path, 'annotations')
    yolo_portrait_ann = os.path.join(portarait_dataset_path, 'yolo_annotations')

    if not os.path.exists(yolo_portrait_ann):
        os.makedirs(yolo_portrait_ann)

    src_ann_count = sum([len(files) for _, _, files in os.walk(portrait_ann)])
    dst_ann_count = sum([len(files) for _, _, files in os.walk(yolo_portrait_ann)])
    if src_ann_count != dst_ann_count:
        convert_ann_to_yolo(portrait_ann, yolo_portrait_ann, folder=True)










if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data', help='path to data')
    args = parser.parse_args()
    main(args)
