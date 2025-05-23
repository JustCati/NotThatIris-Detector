import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor




def _process_file(args):
    folder_path, dst_folder_path, path, CLASSESS, CONVERT_CLASS = args
    try:
        mask = Image.open(os.path.join(folder_path, path))
        mask = np.array(mask)
        img_h, img_w = mask.shape

        bboxes = []
        for label_name in CLASSESS.keys():
            label_val = CLASSESS[label_name]
            positions = np.where(mask == label_val)
            if positions[0].size > 0 and positions[1].size > 0:
                y_min, y_max = positions[0].min(), positions[0].max()
                x_min, x_max = positions[1].min(), positions[1].max()

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                w = x_max - x_min
                h = y_max - y_min

                x_center = x_center / img_w
                y_center = y_center / img_h
                w = w / img_w
                h = h / img_h

                converted_label = CONVERT_CLASS[label_val]
                bboxes.append([converted_label, x_center, y_center, w, h])
        
        if bboxes:
            file_extension = path.split('.')[-1]
            output_path = path.replace(file_extension, 'txt')
            with open(os.path.join(dst_folder_path, output_path), 'w') as f:
                for bbox in bboxes:
                    f.write(' '.join(map(str, bbox)) + '\n')
        return f"Processed {path}"
    except Exception as e:
        return f"Error processing {path}: {e}"


def convert_ann_to_yolo(src_path, dst_path, CLASSESS, CONVERT_CLASS):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for folder in os.listdir(src_path):
        folder_path = os.path.join(src_path, folder)
        dst_folder_path = os.path.join(dst_path, folder)
        if not os.path.exists(dst_folder_path):
            os.makedirs(dst_folder_path)

        print(f'Converting annotations from {folder_path} to YOLO format and saving to {dst_folder_path}')
        
        files_to_process = []
        for path in os.listdir(folder_path):
            files_to_process.append((folder_path, dst_folder_path, path, CLASSESS, CONVERT_CLASS))

        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(_process_file, files_to_process), total=len(files_to_process)))
