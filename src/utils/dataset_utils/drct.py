import os
import shutil
import random
from PIL import Image
import multiprocessing
from pqdm.threads import pqdm


def resize_image(file, dst_path, scale_factor):
    img = Image.open(file)
    h, w = img.size
    new_h, new_w = h // scale_factor, w // scale_factor

    img = img.resize((new_h, new_w), Image.BICUBIC)
    img.save(os.path.join(dst_path, os.path.basename(file)))


def generate_lq_images(src_path, DOWN_SCALE_FACTOR=4):
    if not os.path.exists(src_path):
        raise FileNotFoundError('Source path does not exist')

    dst_path = src_path.replace('hq', 'lq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src_count = sum([len(files) for _, _, files in os.walk(src_path)])
    dst_count = sum([len(files) for _, _, files in os.walk(dst_path)])

    if src_count != dst_count:
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



def copy_image(idx, file, dst_path):
    filename, file_ext = os.path.splitext(file)
    dst_file_path = filename + "_" + str(idx) + file_ext
    shutil.copy(file, os.path.join(dst_path, os.path.basename(dst_file_path)))


def create_dataset(src_path, dst_path, scaling_factor, train_split_ratio=0.8):
    dst_path = os.path.join(dst_path, 'hq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src_file_count = sum([len(files) for _, _, files in os.walk(src_path)]) - 10
    dst_file_count = sum([len(files) for _, _, files in os.walk(dst_path)])

    if src_file_count != dst_file_count:
        print(f"Found {dst_file_count} images in the destination instead of {src_file_count}, copying images to remove people tagging...") 
        shutil.rmtree(dst_path)
        os.makedirs(dst_path)

        cpu_count = multiprocessing.cpu_count()
        all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
        args = [{"idx": idx, 
                 "file": file,
                 "dst_path": dst_path} for idx, file in enumerate(all_files)]
        pqdm(args, copy_image, n_jobs=cpu_count, argument_type="kwargs")

        for file in os.listdir(dst_path):
            if file.startswith("Thumbs"):
                os.remove(os.path.join(dst_path, file))

    generate_lq_images(dst_path, scaling_factor)

    train_path = os.path.join(os.path.dirname(dst_path), 'train')
    dst_file_count = sum([len(files) for _, _, files in os.walk(dst_path)])
    train_count = sum([len(files) for _, _, files in os.walk(train_path)])

    if train_count != (train_split_ratio * dst_file_count):
        print("Splitting dataset")
        split_dataset(dst_path, os.path.dirname(dst_path), train_split_ratio)
