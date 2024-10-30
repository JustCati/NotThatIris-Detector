import os
import shutil
from tqdm import tqdm


def generate_lq_images(src_path):
    pass


def create_dataset(src_path, dst_path):
    dst_path = os.path.join(dst_path, 'hq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src_file_count = sum([len(files) for _, _, files in os.walk(src_path)])
    dst_file_count = sum([len(files) for _, _, files in os.walk(dst_path)])
    print(src_file_count, dst_file_count)

    if src_file_count != dst_file_count:
        shutil.rmtree(dst_path)
        os.makedirs(dst_path)
        all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
        for file in tqdm(all_files):
            dst_file_path = os.path.join(dst_path, os.path.basename(file))

            shutil.copy(file, dst_file_path)
