import os
import argparse

from external.DRCT.drct.archs import *
from src.DRCT.model.DRCT_model import *
from src.DRCT.data.UpsampleDataset_dataset import *
from src.DRCT.losses.WeightedL1Loss_loss import *
from src.DRCT.metrics.WeightedPSNR_metric import *
from src.DRCT.losses.ContextLoss_loss import *

from src.models.yolo import getYOLO
from src.utils.dataset_utils.iris import normalize_dataset, split_by_sample

from basicsr.train import train_pipeline



def main(args):
    root_dir = args.output_path
    dataset_path = args.dataset_path
    root_dir = os.path.join(root_dir, "DRCT")

    test_csv_path = os.path.join(dataset_path, "test_iris.csv")
    train_csv_path = os.path.join(dataset_path, "train_iris.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        split_by_sample(dataset_path)

    if not os.path.exists(os.path.join(dataset_path, "normalized")):
        print("Normalizing iris images...")
        yolo_instance = getYOLO(args.yolo_path, task="segment", device="cuda", inference=True)
        normalize_dataset(yolo_instance, dataset_path)

    train_pipeline(os.path.dirname(__file__))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts"))
    parser.add_argument("--yolo_path", type=str, default="")
    parser.add_argument("-opt", type=str, default=os.path.join(os.path.dirname(__file__), "DRCT.yml")),
    args = parser.parse_args()
    main(args)
