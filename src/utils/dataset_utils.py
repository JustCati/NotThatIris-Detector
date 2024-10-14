import os
import numpy as np
from PIL import Image
from tqdm import tqdm



CLASSESS = {
    "face" : 2,
    "left_eye" : 5,
    "right_eye" : 6
}
CONVERT_CLASS = {
    2 : 1,
    5 : 2,
    6 : 3
}



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
            for label in [CLASSESS["face"], CLASSESS["left_eye"], CLASSESS["right_eye"]]:
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
