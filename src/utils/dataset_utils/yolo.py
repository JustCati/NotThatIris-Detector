import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from tempfile import NamedTemporaryFile


CLASSESS = {
    "left_eye" : 5,
    "right_eye" : 6
}
CONVERT_CLASS = {
    5 : 0,
    6 : 0 # Both eyes are the same class
}




def read_yaml(yaml_path, DATA_PATH, TRAIN_PATH, VAL_PATH):
    easy_portrai_yaml = None
    with open(yaml_path, 'r') as f:
        easy_portrai_yaml = f.read()

    easy_portrai_yaml = easy_portrai_yaml.replace('DATA_PATH', DATA_PATH)
    easy_portrai_yaml = easy_portrai_yaml.replace('TRAIN_PATH', TRAIN_PATH)
    easy_portrai_yaml = easy_portrai_yaml.replace('VAL_PATH', VAL_PATH)

    with NamedTemporaryFile('w', delete=False, suffix=".yaml") as f:
        f.write(easy_portrai_yaml)
        return f.name



def convert_ann_to_yolo(src_path, dst_path):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for folder in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder)
        dst_folder_path = os.path.join(dst_path, folder)
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)

        print(f'Converting annotations from {folder_path} to YOLO format and saving to {dst_folder_path}')
        for path in tqdm(os.listdir(folder_path)):
            mask = Image.open(os.path.join(folder_path, path))
            mask = np.array(mask)
            img_h, img_w = mask.shape

            bboxes = []
            for label in [CLASSESS["left_eye"], CLASSESS["right_eye"]]:
                positions = np.where(mask == label)
                if positions[0].size > 0 and positions[1].size > 0:
                    # Get bounding box
                    y_min, y_max = positions[0].min(), positions[0].max()
                    x_min, x_max = positions[1].min(), positions[1].max()

                    # Convert to yolo format: [class_id, x_center, y_center, w, h]
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    w = x_max - x_min
                    h = y_max - y_min

                    # Normalize
                    x_center = x_center / img_w
                    y_center = y_center / img_h
                    w = w / img_w
                    h = h / img_h

                    label = CONVERT_CLASS[label]
                    bboxes.append([label, x_center, y_center, w, h])
            file_extension = path.split('.')[-1]
            path = path.replace(file_extension, 'txt')
            with open(os.path.join(dst_folder_path, path), 'w') as f:
                for bbox in bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')
