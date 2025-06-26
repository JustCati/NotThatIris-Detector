import torch
import argparse
import tkinter as tk

from src.gui.app import ImageApp
from src.models.yolo import getYOLO
from src.models.mlp_matcher import MLPMatcher
from src.models.backbone import FeatureExtractor
# from src.DRCT.model.DRCT_model import DRCTModel
from src.models.iris_detector import IrisDetector




CLASSES = 99
def main(args):
    backbone = FeatureExtractor(args.backbone)
    backbone.to(args.device)
    backbone.eval()
    
    in_feature = backbone.get_vector_dim()
    
    threshold = torch.load(args.mlp)["threshold"]
    mlp = MLPMatcher.load_from_checkpoint(args.mlp, in_feature=in_feature, num_classes=CLASSES)
    mlp.set_threshold(threshold)
    mlp.to(args.device)
    mlp.eval()
    
    yolo_det = getYOLO(args.yolo_det, task='detection', device=args.device, inference=True)
    yolo_det.to(args.device)
    yolo_det.eval()
    
    yolo_seg = getYOLO(args.yolo_seg, task='segment', device=args.device, inference=True)
    yolo_seg.to(args.device)
    yolo_seg.eval()
    
    
    iris_detector = IrisDetector(modules={
        "yolo_det": yolo_det,
        "yolo_seg": yolo_seg,
        "sr": None,  # Placeholder for super-resolution model if needed
        "backbone": backbone,
        "mlp": mlp,
    })
    iris_detector.to(args.device)
    

    with open("label_map.txt", "r") as f:
        label_map = [line.strip() for line in f.readlines()]
        label_map = {int(elem.split(": ")[0]): int(elem.split(": ")[1]) for elem in label_map}
    
    root = tk.Tk()
    _ = ImageApp(root, model=iris_detector, label_map=label_map, csv=args.csv_path, th=threshold)
    root.mainloop()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Detector Application")
    parser.add_argument("--backbone", type=str, default="", help="Path to the backbone model", required=True)
    parser.add_argument("--yolo_det", type=str, default="", help="Path to the YOLO detection model", required=True)
    parser.add_argument("--yolo_seg", type=str, default="", help="Path to the YOLO segmentation model", required=False)
    parser.add_argument("--sr", type=str, default="", help="Path to the super-resolution model", required=False)
    parser.add_argument("--mlp", type=str, default="", help="Path to the MLP model", required=True)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--csv_path", type=str, default="", help="Path to the annotation csv")
    args = parser.parse_args()
    main(args)
