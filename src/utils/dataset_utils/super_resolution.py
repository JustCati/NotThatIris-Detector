import os
import cv2 
import shutil
import random
from PIL import Image
import multiprocessing
from pqdm.threads import pqdm
from torchvision.transforms import v2 as T

from src.utils.eyes import iris_hough_detector



def resize_image(file, dst_path, scale_factor):
    img = Image.open(file)
    pipeline = T.Compose([
        T.Resize((img.size[1] // scale_factor, img.size[0] // scale_factor), interpolation=Image.BICUBIC),
        T.GaussianBlur(5, 1.5),
        T.JPEG(quality=(25, 40)),
    ])
    try:
        img = pipeline(img)
        img.save(os.path.join(dst_path, os.path.basename(file)))
    except:
        os.remove(file)


def generate_lq_images(src_path, DOWN_SCALE_FACTOR=4):
    if not os.path.exists(src_path):
        raise FileNotFoundError('Source path does not exist')

    dst_path = src_path.replace('hq', 'lq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    print("Generating low quality images...")
    shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    cpu_count = multiprocessing.cpu_count()
    all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
    args = [{
        "file": file,
        "dst_path": dst_path,
        "scale_factor": DOWN_SCALE_FACTOR} for file in all_files]
    pqdm(args, resize_image, n_jobs=cpu_count, argument_type="kwargs")


def split_single_image(idx, file, train_hq_path, val_hq_path, train_lq_path, val_lq_path, train_idx):
    if idx in train_idx:
        shutil.copy(file, os.path.join(train_hq_path, os.path.basename(file)))
        shutil.copy(file.replace('hq', 'lq'), os.path.join(train_lq_path, os.path.basename(file)))
    else:
        shutil.copy(file, os.path.join(val_hq_path, os.path.basename(file)))
        shutil.copy(file.replace('hq', 'lq'), os.path.join(val_lq_path, os.path.basename(file)))


def split_dataset(src_path, dst_path, split_ratio=0.8):
    if not os.path.exists(src_path):
        raise FileNotFoundError('Source path does not exist')

    train_path = os.path.join(dst_path, 'train')
    val_path = os.path.join(dst_path, 'val')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    train_hq_path = os.path.join(train_path, 'hq')
    train_lq_path = os.path.join(train_path, 'lq')
    val_hq_path = os.path.join(val_path, 'hq')
    val_lq_path = os.path.join(val_path, 'lq')
    if not os.path.exists(train_hq_path):
        os.makedirs(train_hq_path)
    if not os.path.exists(val_hq_path):
        os.makedirs(val_hq_path)
    if not os.path.exists(train_lq_path):
        os.makedirs(train_lq_path)
    if not os.path.exists(val_lq_path):
        os.makedirs(val_lq_path)

    src_count = sum([len(files) for _, _, files in os.walk(src_path)])
    train_count = int(src_count * split_ratio)

    indexis = list(range(src_count))
    random.seed(42)
    random.shuffle(indexis)
    train_idx = set(indexis[:train_count])

    cpu_count = multiprocessing.cpu_count()
    all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
    args = [{"idx": idx, "file": file, 
             "train_hq_path": train_hq_path,
             "val_hq_path": val_hq_path,
             "train_lq_path": train_lq_path,
             "val_lq_path": val_lq_path,
             "train_idx": train_idx}  for idx, file in enumerate(all_files)]
    pqdm(args, split_single_image, n_jobs=cpu_count, argument_type="kwargs")



def copy_image(idx, file, dst_path, scaling_factor):
    filename, file_ext = os.path.splitext(file)
    dst_file_path = filename + "_" + str(idx) + file_ext

    img = cv2.imread(file)
    if img is None:
        return

    image_roi, _, success = iris_hough_detector(file)
    if success:
        im = Image.fromarray(image_roi)

        if im.size[0] % scaling_factor != 0 or im.size[1] % scaling_factor != 0:
            im = im.crop((0, 0, im.size[0] - im.size[0] % scaling_factor, im.size[1] - im.size[1] % scaling_factor))
        if (im.size[0] / scaling_factor) < 64 or (im.size[1] / scaling_factor) < 64:
            im = im.resize((64 * scaling_factor, 64 * scaling_factor), Image.BICUBIC)

        try:
            im.save(os.path.join(dst_path, os.path.basename(dst_file_path)))
        except:
            return


def create_dataset(src_path, dst_path, scaling_factor, train_split_ratio=0.8):
    dst_path = os.path.join(dst_path, 'hq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    print("Copying high quality images...")
    shutil.rmtree(dst_path)
    os.makedirs(dst_path)

    cpu_count = multiprocessing.cpu_count()
    all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
    args = [{"idx": idx, 
                "file": file,
                "dst_path": dst_path,
                "scaling_factor": scaling_factor} for idx, file in enumerate(all_files)]
    pqdm(args, copy_image, n_jobs=cpu_count, argument_type="kwargs")

    for file in os.listdir(dst_path):
        if file.startswith("Thumbs"):
            os.remove(os.path.join(dst_path, file))

    generate_lq_images(dst_path, scaling_factor)

    train_path = os.path.join(os.path.dirname(dst_path), 'train')
    dst_file_count = sum([len(files) for _, _, files in os.walk(dst_path)])
    train_count = sum([len(files) for _, _, files in os.walk(train_path)]) // 2

    if train_count != (int(train_split_ratio * dst_file_count)):
        print(f"Found {train_count} images in the train folder instead of {int(train_split_ratio * dst_file_count)}, splitting dataset...")
        split_dataset(dst_path, os.path.dirname(dst_path), train_split_ratio)
