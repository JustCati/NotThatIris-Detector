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
    2 : 0,
    5 : 1,
    6 : 2
}



def convert_ann_to_yolo(src_path, dst_path, folder = True):
    if not os.path.exists(dst_path) and folder:
        os.makedirs(dst_path)

    if folder:
        for path in os.listdir(src_path):
            if not os.path.isfile(path):
                if os.path.isfile(os.path.join(src_path, path)):
                    folder = False
                convert_ann_to_yolo(os.path.join(src_path, path), os.path.join(dst_path, path), folder)
    else:
        src_path = os.path.dirname(src_path)
        dst_path = os.path.dirname(dst_path)

        print(f'Converting annotations from {src_path} to YOLO format and saving to {dst_path}')
        for path in tqdm(os.listdir(src_path)):
            mask = Image.open(os.path.join(src_path, path))
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
            with open(os.path.join(dst_path, path), 'w') as f:
                for bbox in bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')
