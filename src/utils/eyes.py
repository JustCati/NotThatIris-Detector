import numpy as np
from PIL import Image
from src.models.yolo import detect



def get_irismask(eyeimage, model, output_size=(128, 128)):
    res = detect(model, eyeimage, device="cuda")
    pred_classes = res.boxes.cls.cpu().numpy()

    iris_idx = np.where(pred_classes == 0)[0][0]
    iris_box = res.boxes[iris_idx].xyxy.cpu().numpy().squeeze()
    
    iris_mask = res.masks[iris_idx].data.cpu().numpy().squeeze()
    iris_mask = Image.fromarray(iris_mask)
    iris_mask = iris_mask.resize(eyeimage.size, Image.NEAREST)
    iris_mask = np.array(iris_mask)

    iris_image = eyeimage.copy()
    iris_image = np.array(iris_image)
    iris_image[iris_mask == 0] = 0  # Set background to black
    iris_image = Image.fromarray(iris_image)
    iris_image = iris_image.crop(iris_box)
    iris_image = iris_image.resize(output_size, Image.BILINEAR)
    iris_image = np.array(iris_image)
    return iris_image
