import cv2
import numpy as np
from PIL import Image




def daugman_normalizaiton(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)
    r_out = r_in + r_out

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


def recflection_remove(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    ret, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    dst = cv2.inpaint(img, dilation, 5, cv2.INPAINT_TELEA)
    return dst


def iris_hough_detector(image_path, r = 40):
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
            image = image[i[1] - i[2] - r:i[1] + i[2] + r, i[0] - i[2] -r:i[0] + i[2] + r]
            radius = i[2]
        success = True
        return image, radius, success
    except:
        image[:] = 255
        success = False
        return image, image.shape[0], success



def normalize_eye(image, radius_out=60, find_iris=True):
    if isinstance(image, str):
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    success = False
    if find_iris:
        image_roi, radius, success = iris_hough_detector(image, 40)
    if success or not find_iris:
        image_roi = image if not find_iris else image_roi
        image_roi = recflection_remove(image_roi)
        normalized = daugman_normalizaiton(image_roi, 60, 360, radius if success else radius_out, radius_out)
        return normalized
    else:
        raise ValueError("No iris found in the image.")
