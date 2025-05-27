import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import cv2
import random
import configparser
from torchvision.transforms import v2 as T

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

        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            results = list(tqdm(executor.map(_process_file, files_to_process), total=len(files_to_process)))


def convert_ann_to_seg(ann_path, out_path, classes=3):
    def reduce(img_path):
        img = Image.open(img_path)
        img = np.array(img)
        img[img == 255] = 1
        img = Image.fromarray(img.astype(np.uint8))
        img.save(img_path)
    
    for folder in os.listdir(ann_path):
        folder_path = os.path.join(ann_path, folder)
        out_path = os.path.join(os.path.dirname(ann_path), "labels", folder)
        os.makedirs(out_path, exist_ok=True)

        files_to_process = []
        for file in os.listdir(folder_path):
            if file.endswith('.png'):
                files_to_process.append(os.path.join(folder_path, file))
        
        print("Normalizing values in masks...")
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            list(tqdm(executor.map(lambda x: reduce(x), files_to_process), total=len(files_to_process)))

        convert_segment_masks_to_yolo_seg(folder_path, out_path, classes=3)


def process_file(in_path, out_path):
    pipeline = T.Compose([
        T.GaussianBlur(kernel_size=15, sigma=(1, 2)),
        T.JPEG(quality=25),
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
    with ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(lambda x: process_file(*x), files_to_process), total=len(files_to_process)))



def create_iris_pupil_segmentation_mask(iris_mask,  pupil_params):
    iris_mask[iris_mask == 255] = 1
    mask_pupil = np.zeros(iris_mask.shape, dtype=np.uint8)

    mask_pupil = cv2.circle(
        mask_pupil,
        center=tuple(map(int, pupil_params['center'])),
        radius=int(pupil_params['radius']),
        color=2,
        thickness=-1
    )
    
    ys, xs = np.where(iris_mask == 1)
    if len(xs) == 0 or len(ys) == 0:
        print("No iris pixels found in the mask.")
        combined_mask = iris_mask.copy()
        combined_mask[mask_pupil == 2] = 2
        return combined_mask

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    mask_pupil[:y_min, :] = 0
    mask_pupil[y_max+1:, :] = 0
    mask_pupil[:, :x_min] = 0
    mask_pupil[:, x_max+1:] = 0

    combined_mask = iris_mask.copy()
    combined_mask[mask_pupil == 2] = 2
    return combined_mask



def process_mask_file(mask_file, localization_filepath):
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    config = configparser.ConfigParser()
    config.read(localization_filepath)
    pupil_params = {
        'center': (float(config['pupil']['center_x']), float(config['pupil']['center_y'])),
        'radius': float(config['pupil']['radius'])
    }

    final_mask = create_iris_pupil_segmentation_mask(mask, pupil_params)
    final_mask = Image.fromarray(final_mask)
    output_mask_path = mask_file.replace("masks", "annotations")
    os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
    final_mask.save(output_mask_path)



def generate_annotations(dataset_path):
    mask_path = os.path.join(dataset_path, 'masks')
    localization_path = os.path.join(dataset_path, 'localization')
    
    file_to_process = []
    for folder in os.listdir(mask_path):
        folder_path = os.path.join(mask_path, folder)
        for mask_file in os.listdir(folder_path):
            if mask_file.endswith('.png'):
                file_to_process.append((os.path.join(folder_path, mask_file),
                                        os.path.join(localization_path, mask_file.replace('.png', '.ini'))))
# multiprocessing.cpu_count()
    print(f"Found {len(file_to_process)} mask files to process in {mask_path}")
    with ThreadPoolExecutor(max_workers=1) as executor:
        list(tqdm(executor.map(lambda x: process_mask_file(*x), file_to_process), total=len(file_to_process)))


def split_data(dataset_path, split_ratio=0.8):
    random.seed(42)

    images_path = os.path.join(dataset_path, 'images_raw')
    all_images_path = random.sample(os.listdir(images_path), len(os.listdir(images_path)))
    split_index = int(len(os.listdir(images_path)) * split_ratio)
    train_images = all_images_path[:split_index]
    test_images = all_images_path[split_index:]
    
    train_img_path = os.path.join(dataset_path, "images_raw", 'train')
    test_img_path = os.path.join(dataset_path, "images_raw", 'test')
    os.makedirs(train_img_path, exist_ok=True)
    os.makedirs(test_img_path, exist_ok=True)

    train_masks_path = os.path.join(dataset_path, "masks", 'train')
    test_masks_path = os.path.join(dataset_path, "masks", 'test')
    os.makedirs(train_masks_path, exist_ok=True)
    os.makedirs(test_masks_path, exist_ok=True)

    for img in train_images:
        src_img = os.path.join(images_path, img)
        dst_img = os.path.join(train_img_path, img)
        if os.path.exists(src_img):
            os.rename(src_img, dst_img)

        src_mask = os.path.join(dataset_path, 'masks', img.replace('.jpg', '.png'))
        dst_mask = os.path.join(train_masks_path, img.replace('.jpg', '.png'))
        if os.path.exists(src_mask):
            os.rename(src_mask, dst_mask)

    for img in test_images:
        src_img = os.path.join(images_path, img)
        dst_img = os.path.join(test_img_path, img)
        if os.path.exists(src_img):
            os.rename(src_img, dst_img)

        src_mask = os.path.join(dataset_path, 'masks', img.replace('.jpg', '.png'))
        dst_mask = os.path.join(test_masks_path, img.replace('.jpg', '.png'))
        if os.path.exists(src_mask):
            os.rename(src_mask, dst_mask)
