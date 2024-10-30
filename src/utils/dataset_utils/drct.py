import os
import shutil
from PIL import Image
from tqdm import tqdm




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

        all_files = [os.path.join(root, file) for root, _, files in os.walk(src_path) for file in files]
        for file in tqdm(all_files):
            img = Image.open(file)
            h, w = img.size
            new_h, new_w = h // DOWN_SCALE_FACTOR, w // DOWN_SCALE_FACTOR

            img = img.resize((new_h, new_w), Image.BICUBIC)
            img.save(os.path.join(dst_path, os.path.basename(file)))





def create_dataset(src_path, dst_path):   
    dst_path = os.path.join(dst_path, 'hq')
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    src_file_count = sum([len(files) for _, _, files in os.walk(src_path)]) - 10
    dst_file_count = sum([len(files) for _, _, files in os.walk(dst_path)])

    if src_file_count != dst_file_count:
        print(f"Found {dst_file_count} images in the destination instead of {src_file_count}, copying images to remove people tagging...") 
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
