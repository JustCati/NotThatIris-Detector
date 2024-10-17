import os
import numpy as np
from PIL import Image



def get_medium_eye_size(ann_path, img_path):
    if not os.path.exists(ann_path):
        raise ValueError('Annotation path does not exist')

    widths, heights = [], []
    medium_eye_size = []
    files = [file for file in os.listdir(ann_path) if not file.endswith('.cache')]
    for group in files:
        for file in os.listdir(os.path.join(ann_path, group)):
            mask = Image.open(os.path.join(img_path, group, file.replace('txt', 'jpg')))
            mask = np.array(mask)
            img_h, img_w, _ = mask.shape
            with open(os.path.join(ann_path, group, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    label, x_center, y_center, w, h = map(float, line.split())
                    x_center, y_center = x_center * img_w, y_center * img_h
                    w, h = w * img_w, h * img_h
                    widths.append(w)
                    heights.append(h)
                    if label == 0:
                        continue
                    x_min, x_max, y_min, y_max = x_center - w/2, x_center + w/2, y_center - h/2, y_center + h/2
                    medium_eye_size.append((x_max - x_min) * (y_max - y_min))
    with open("stats.txt", "w") as f:
        f.write(f"AREA -> Median: {np.median(medium_eye_size)} Mean: {np.mean(medium_eye_size)}")
        f.write(f"\n\nWidths -> Median: {np.median(widths)} Mean: {np.mean(widths)}")
        f.write(f"\n\nHeights -> Median: {np.median(heights)} Mean: {np.mean(heights)}")



if __name__ == "__main__":
    portrait_dataset_path = os.path.join(os.getcwd(), "data", "EasyPortrait", "labels")
    portrait_image_path = os.path.join(os.getcwd(), "data", "EasyPortrait", "images")
    get_medium_eye_size(portrait_dataset_path, portrait_image_path)

