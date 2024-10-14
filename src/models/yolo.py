import os
import torch
from ultralytics import YOLO



def getYOLO(checkpoint_path: str, device: str = 'cpu') -> YOLO:
    if not os.path.exists(checkpoint_path):
        raise ValueError('Checkpoint path does not exist')
    model = YOLO(checkpoint_path)
    model.to(device)
    return model
