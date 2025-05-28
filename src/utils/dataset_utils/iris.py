import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.models.yolo import detect
from src.utils.eyes import get_irismask



def containment_percentage(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    area_a = (xa2 - xa1) * (ya2 - ya1)

    if area_a == 0:
        return 0.0
    return inter_area / area_a


def insert_eye(eyes, box, side):
    if len(eyes[side]) > 0:
        if containment_percentage(box, eyes[side][-1]) < 0.1:
            eyes[side].append(box)
            return
        return
    eyes[side].append(box)


def find_eyes(yolo_det, img, PADDING=75):
    results = detect(yolo_det, img, device="cuda")
    
    eyes = {
        0: [],
        1: []
    }
    half_width = img.width // 2

    for result in results:
        res = result.boxes
        box = res.xyxy[0].cpu().numpy()
        side = 0 if box[0] < half_width else 1
        insert_eye(eyes, box, side)

    for k, v in eyes.items():
        if len(v) == 0:
            continue
        x1, y1, x2, y2 = v[0]
        x1 = max(0, x1 - PADDING)
        y1 = max(0, y1 - PADDING)
        x2 = min(img.width, x2 + PADDING)
        y2 = min(img.height, y2 + PADDING)
        eyes[k] = [x1, y1, x2, y2]

    output = []
    for k, v in eyes.items():
        if len(v) == 0:
            continue
        x1, y1, x2, y2 = v
        crop = img.crop((x1, y1, x2, y2))
        output.append(crop)
    return output




def normalize_dataset(yolo_instance, dataset_path, save_masks=False, distance=False): 
    all_image_full_paths = []

    for root_dir, _, file_names in os.walk(dataset_path):
        for file_name in file_names:
            if file_name.endswith(".jpg"):
                all_image_full_paths.append(os.path.join(root_dir, file_name))

    if not all_image_full_paths:
        print(f"No .jpg files found in {dataset_path}")
        return

    for input_image_path in tqdm(all_image_full_paths, desc="Normalizing images"):
        image = Image.open(input_image_path).convert("RGB")
        if distance:
            yolo_det_istance, yolo_seg_instance = yolo_instance
            eyes = find_eyes(yolo_det_istance, image)
            eyes = [(img, input_image_path.replace(".jpg", f"_{i}.jpg")) for i, img in enumerate(eyes)]
        else:
            yolo_seg_instance = yolo_instance
            eyes = [(image, input_image_path)]

        for eye in eyes:
            image, input_image_path = eye
            try:
                norm = get_irismask(image, yolo_seg_instance)
                if norm is None:
                    print(f"Normalization failed for {input_image_path}, skipping.")
                    continue

                output_image_path = input_image_path.replace("images_raw", "normalized")
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                cv2.imwrite(output_image_path, norm)
                if save_masks:
                    mask = norm.copy()
                    mask = np.array(mask)
                    mask[mask > 0] = 1

                    masks_output_path = input_image_path.replace("images_raw", "masks")
                    os.makedirs(os.path.dirname(masks_output_path), exist_ok=True)
                    cv2.imwrite(masks_output_path, mask)
            except Exception as e:
                print(f"Error saving image {output_image_path}: {e}")
                continue


def build_df(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    if not os.path.basename(dataset_path) == "images":
        img_path = os.path.join(dataset_path, "images")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Dataset image path {img_path} does not exist.")

    data = []
    for user in os.listdir(img_path):
        user_path = os.path.join(img_path, user)
        for side in sorted(os.listdir(user_path)):
            side_path = os.path.join(user_path, side)
            for img in os.listdir(side_path):
                if not img.endswith(".jpg"):
                    continue
                user_id = f"{img[2:5]}-{img[5]}"
                data.append(
                    {
                        "Label": user_id,
                        "ImagePath": os.path.join(side_path, img),
                    }
                )
    df = pd.DataFrame(data)
    return df


def split_by_sample(dataset_path, train_ration=0.8):
    df = build_df(dataset_path)
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    train_df = df.iloc[:int(len(df)*train_ration)]
    test_df = df.iloc[int(len(df)*train_ration):]

    assert len(train_df) + len(test_df) == len(df)
    assert train_df.index.intersection(test_df.index).empty

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(os.path.join(dataset_path, "train_iris.csv"))
    test_df.to_csv(os.path.join(dataset_path, "test_iris.csv"))







# def split_iris_thousand_users(csv_path,
#                               known_ratio=0.8):
#     random.seed(4242)
#     df = pd.read_csv(csv_path, index_col=0)
#     users = df["Label"].apply(lambda x: x.split("-")[0]).unique()

#     random.shuffle(users)
#     known_users = users[:int(len(users)*known_ratio)]
#     unknown_users = users[int(len(users)*known_ratio):]

#     train_df = df[df["Label"].apply(lambda x: x.split("-")[0] in known_users)]

#     unknown_users_train = unknown_users[:int(len(unknown_users)*0.6)]
#     unknown_users_test = unknown_users[int(len(unknown_users)*0.6):int(len(unknown_users)*0.8)]
#     unknown_users_val = unknown_users[int(len(unknown_users)*0.8):]

#     unknown_trainDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_train)]
#     unknown_testDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_test)]
#     unknown_valDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_val)]

#     known_trainDF = train_df.sample(frac=0.8, random_state=4242)
#     known_testDF = train_df.drop(known_trainDF.index).sample(frac=0.1, random_state=4242)
#     known_valDF = train_df.drop(known_trainDF.index).drop(known_testDF.index)

#     assert set(unknown_testDF.Label.unique()).intersection(set(known_trainDF.Label.unique())) == set()
#     assert set(unknown_testDF.Label.unique()).intersection(set(known_testDF.Label.unique())) == set()
#     assert set(unknown_testDF.Label.unique()).intersection(set(known_valDF.Label.unique())) == set()

#     assert set(unknown_testDF.Label.unique()).intersection(set(unknown_valDF.Label.unique())) == set()

#     unknown_testDF["Label"] = -1
#     unknown_valDF["Label"] = -1
#     unknown_trainDF["Label"] = -1

#     train_df = pd.concat([known_trainDF, unknown_trainDF])
#     test_df = pd.concat([known_testDF, unknown_testDF])
#     val_df = pd.concat([known_valDF, unknown_valDF])

#     train_df.reset_index(drop=True, inplace=True)
#     test_df.reset_index(drop=True, inplace=True)
#     val_df.reset_index(drop=True, inplace=True)

#     train_df.to_csv(csv_path.replace(os.path.basename(csv_path), "train_users.csv"))
#     test_df.to_csv(csv_path.replace(os.path.basename(csv_path), "test_users.csv"))
#     val_df.to_csv(csv_path.replace(os.path.basename(csv_path), "val_users.csv"))
