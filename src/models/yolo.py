import os
import torch
from ultralytics import YOLOv10



def getYOLO(checkpoint_path: str, device: str = 'cpu') -> YOLOv10:
    if not os.path.exists(checkpoint_path):
        raise ValueError('Checkpoint path does not exist')
    model = YOLOv10(checkpoint_path)
    model.to(device)
    return model
