import os
import torch
import subprocess
import multiprocessing
from ultralytics import YOLOv10 as YOLO



def getYOLO(checkpoint_path: str, device: str = 'cpu', inference: bool = False):
    download = False
    if not os.path.exists(checkpoint_path):
        download = True
    if download:
        link = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt"
        model_path = os.path.basename(link)
        if not os.path.exists(model_path):
            print("Downloading YOLO model...")
            subprocess.run(["wget", link])
            print("Download complete.")
        model = YOLO(model_path, task='detect')
    else:
        model = YOLO(checkpoint_path, task='detect')
    if not inference:
        model.to(device)
    return model


def train(model: YOLO, 
          yaml_file: str,
          epochs: int,
          patience: int,
          batch_size: int,
          model_path: str,
          folder_name: str,
          resume: bool = False,
          device: str = 'cuda'):
    cpu_workers = multiprocessing.cpu_count()
    results = model.train(data=yaml_file, 
                          batch=batch_size,
                          imgsz=640,
                          epochs=epochs if patience == 0 else 100,
                          verbose=True,
                          patience=patience,
                          workers=cpu_workers,
                          save_period=5,
                          device=device,
                          resume=resume,
                          val=True,
                          plots=True,
                          project = model_path,
                          name = folder_name)
    return results
