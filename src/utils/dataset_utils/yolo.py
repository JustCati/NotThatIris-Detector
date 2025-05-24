import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ultralytics.data.converter import convert_segment_masks_to_yolo_seg



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







def convert_ann_to_seg(ann_path, out_path, classes=3):
    for folder in os.listdir(ann_path):
        folder_path = os.path.join(ann_path, folder)
        out_path = os.path.join(os.path.dirname(ann_path), "labels", folder)
        os.makedirs(out_path, exist_ok=True)
        convert_segment_masks_to_yolo_seg(folder_path, out_path, classes=3)

    out_path = os.path.join(os.path.dirname(ann_path), "labels")

    for folder in os.listdir(out_path):
        folder_path = os.path.join(out_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            with open(os.path.join(folder_path, file), "r") as f:
                lines = f.readlines()

            classes = {
                1: [],
                2: [],
            }
            for line in lines:
                line = line.strip().split()
                class_id = int(line[0])
                if class_id == 0:
                    continue
                points = list(map(float, line[1:]))
                points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
                if len(classes[class_id]) < 1:
                    classes[class_id].append(points)
                if len(classes[class_id]) == 1:
                    actual = len(classes[class_id][0])
                    if actual < len(points):
                        classes[class_id][0] = points
            with open(os.path.join(folder_path, file), "w") as f:
                for class_id, points in classes.items():
                    class_id -= 1
                    if len(points) > 0:
                        points = points[0]
                        points_str = " ".join(f"{x:.6f}" for point in points for x in point)
                        f.write(f"{class_id} {points_str}\n")