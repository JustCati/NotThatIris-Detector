import os
import torch 
import argparse
import pandas as pd
import multiprocessing
from torch.utils.data import DataLoader

import lightning as L
from torchvision.transforms import v2 as T
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.models.resnet import Resnet
from src.models.yolo import getYOLO
from src.dataset.NormalizedDataset import NormalizedIrisDataset
from src.utils.dataset_utils.iris import normalize_dataset, split_by_sample




def main(args):
    dataset_path = args.dataset_path
    root_dir = args.output_path
    root_dir = os.path.join(root_dir, "RESNET50")

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    test_csv_path = os.path.join(dataset_path, "test_iris.csv")
    train_csv_path = os.path.join(dataset_path, "train_iris.csv")
    
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        split_by_sample(dataset_path)

    if not os.path.exists(os.path.join(dataset_path, "normalized")):
        print("Normalizing iris images...")
        yolo_instance = getYOLO(args.yolo_path, task="segment", device="cuda", inference=True)
        normalize_dataset(yolo_instance, dataset_path)

    transform = T.Compose([
        T.GaussianBlur(kernel_size=3),
        T.JPEG(quality=(50, 75)),
    ])

    train_df = pd.read_csv(train_csv_path, index_col=0)
    test_df = pd.read_csv(test_csv_path, index_col=0)
    train_users = train_df["Label"].unique()
    test_users = test_df["Label"].unique()
    all_users = list(set(train_users).union(set(test_users)))

    cpu_count = multiprocessing.cpu_count() // 2
    train_dataset = NormalizedIrisDataset(train_csv_path, all_users, transform=transform)
    eval_dataset = NormalizedIrisDataset(test_csv_path, all_users, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count)

    model = Resnet(num_classes=train_dataset.num_classes, batch_size=args.batch_size, verbose=True)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="resnet")
    tb_logger = TensorBoardLogger(os.path.join(root_dir, "logs"), name="resnet", version=csv_logger.version)

    best_checkpoint_saver = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "models"),
        filename="best",
        save_top_k=1,
        monitor="eval/f1",
        mode="max",
        verbose=True,
        save_last=True
        )

    early_stop_callback = EarlyStopping(
        monitor="eval/f1",
        min_delta=0.006,
        patience=5,
        verbose=False, 
        mode="max"
        )

    trainer = L.Trainer(
        default_root_dir=root_dir,   
        max_epochs=args.num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=[csv_logger, tb_logger],
        callbacks=[best_checkpoint_saver, early_stop_callback]
        )

    trainer.fit(model=model, 
                train_dataloaders=train_dataloader,
                val_dataloaders=eval_dataloader
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Lamp"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts"))
    parser.add_argument("--yolo_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
