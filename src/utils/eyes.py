import cv2
import torch
import numpy as np
from src.models.yolo import detect



def daugman_normalization(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)

    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]
            r_pro = j / height

            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]

            flat[j][i] = color
    return flat



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



def normalize_eye(image, model, output_size=(60, 360)):
    detections = {
        0: [], # Pupils
        1: [], # Iris
    }
    results = detect(model, image)

    for result in results:
        class_id = int(result.boxes.cls.cpu().numpy()[0])
        if class_id != 1:
            continue
        box = result.boxes.xyxy[0].cpu().numpy()
        score = result.boxes.conf.cpu().numpy()[0]

        if len(detections[1]) == 1:
            actual_score = detections[1][0][2]
            if score > actual_score:
                detections[1][0] = (box, score)
        else:
            detections[1].append((box, score))
            

    for result in results:
        class_id = int(result.boxes.cls.cpu().numpy()[0])
        if class_id != 0:
            continue
        box = result.boxes.xyxy[0].cpu().numpy()
        score = result.boxes.conf.cpu().numpy()[0]
        
        if len(detections[0]) == 1:
            iris_box, _ = detections[1][0]
            iou = containment_percentage(box, iris_box)
            if iou >= 0.8:
                actual_score = detections[0][0][1]
                if score > actual_score:
                    detections[0][0] = (box, score)
        else:
            iris_box, _ = detections[1][0]
            iou = containment_percentage(box, iris_box)
            if iou >= 0.8:
                detections[0].append((box, score))

    iris_box = detections[1][0][0] if len(detections[1]) > 0 else None
    pupil_box = detections[0][0][0] if len(detections[0]) > 0 else None
    if iris_box is None or pupil_box is None:
        print("No iris or pupil detected")
        return None
    image_crop = image.crop(
        (
            int(iris_box[0]),
            int(iris_box[1]),
            int(iris_box[2]),
            int(iris_box[3])
        )
    )
    image_crop = np.array(image_crop)
    image_r = image_crop.shape[0] / 2
    pupil_r = int((pupil_box[2] - pupil_box[0]) / 2)

    normalized_eye = daugman_normalization(image_crop, output_size[0], output_size[1], pupil_r, image_r)
    return normalized_eye
