import os
import multiprocessing
from ultralytics import YOLOv10



def getYOLO(checkpoint_path: str, device: str = 'cpu') -> YOLOv10:
    if not os.path.exists(checkpoint_path):
        raise ValueError('Checkpoint path does not exist')
    model = YOLOv10(checkpoint_path)
    model.to(device)
    return model


def train(model: YOLOv10, yaml_file: str, epochs: int, batch_size: int, model_path: str, device: str = 'cuda'):
    cpu_workers = multiprocessing.cpu_count()
    results = model.train(data=yaml_file, 
                          batch=batch_size,
                          imgsz=1280,
                          epochs=epochs,
                          verbose=True,
                          workers=cpu_workers,
                          save_period=1,
                          device=device,
                          val=True,
                          plots=True,
                          project = model_path,
                          name = "YOLOv10")
    return results
