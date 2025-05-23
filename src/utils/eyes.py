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


def hough_detector(image_path, padding_r=0):
    success = False
    image = cv2.imread(image_path) if isinstance(image_path, str) else cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(image, 11)
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        1,
        50,
        param1=ret,
        param2=30,
        minRadius=20,
        maxRadius=100,
    )
    try:
        circles = circles[0, :, :]
        circles = np.int16(np.array(circles))
        for i in circles[:]:
            image = image[i[1] - i[2] - padding_r:i[1] + i[2] + padding_r, i[0] - i[2] - padding_r:i[0] + i[2] + padding_r]
            radius = i[2]
        success = True
        return image, radius, success
    except:
        image[:] = 255
        success = False
        return image, image.shape[0], success



def find_pupil(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_roi, radius, success = hough_detector(image, -5)
    return image_roi, radius, success



def normalize_eye(image, model, output_size=(60, 360)):
    results = detect(model, image, "cuda" if torch.cuda.is_available() else "cpu")
    if len(results) == 0:
        raise ValueError("No eye detected in the image.")
    try:
        box = results.boxes.xyxy.cpu().tolist()[0]
    except:
        print("No eye detected in the image. Returning a blank image.")
        return np.zeros((output_size[0], output_size[1], 3), dtype=np.uint8)

    image_crop = image.crop(
        (
            box[0],
            box[1],
            box[2],
            box[3],
        )
    )
    image_crop = np.array(image_crop)
    image_r = image_crop.shape[0] // 2

    _, pupil_r, success = find_pupil(image_crop)
    if not success:
        pupil_r = image_r // 2

    normalized_eye = daugman_normalization(image_crop, output_size[0], output_size[1], pupil_r, image_r)
    return normalized_eye

