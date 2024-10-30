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

    if src_file_count != dst_file_count:
        shutil.rmtree(dst_path)
        os.makedirs(dst_path)
        all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
        for idx, file in enumerate(tqdm(all_files)):
            filename, file_ext = os.path.splitext(file)
            dst_file_path = filename + "_" + str(idx) + file_ext
            shutil.copy(file, os.path.join(dst_path, os.path.basename(dst_file_path)))
        for file in os.listdir(dst_path):
            if file.startswith("Thumbs"):
                os.remove(os.path.join(dst_path, file))
    return dst_path
