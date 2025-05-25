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

from src.models.dncnn import DNCNN
from src.models.yolo import getYOLOseg
from src.dataset.DenoiseDataset import NormalizedIrisDataset
from src.utils.dataset_utils.iris import normalize_dataset, split_by_sample




def main(args):
    dataset_path = args.dataset_path
    root_dir = args.output_path
    root_dir = os.path.join(root_dir, "DNCNN")

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    images_path = os.path.join(dataset_path, "normalized")
    test_csv_path = os.path.join(dataset_path, "test_iris.csv")
    train_csv_path = os.path.join(dataset_path, "train_iris.csv")

    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        split_by_sample(dataset_path)

    if not os.path.exists(os.path.join(dataset_path, "normalized")):
        print("Normalizing iris images...")
        yolo_instance = getYOLOseg(args.yolo_path, device="cuda", inference=True)
        normalize_dataset(yolo_instance, dataset_path)

    transform = T.Compose([
        T.GaussianBlur(kernel_size=15, sigma=1.5),
        T.JPEG(quality=(20, 50)),
    ])

    cpu_count = multiprocessing.cpu_count() // 2
    train_dataset = NormalizedIrisDataset(train_csv_path, transform=transform)
    eval_dataset = NormalizedIrisDataset(test_csv_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count)

    model = DNCNN(verbose=True)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="iris-thousand")
    tb_logger = TensorBoardLogger(os.path.join(root_dir, "logs"), name="iris-thousand", version=csv_logger.version)

    best_checkpoint_saver = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "models"),
        filename="best",
        save_top_k=1,
        monitor="eval/psnr",
        mode="max",
        verbose=True,
        save_last=True
        )

    early_stop_callback = EarlyStopping(
        monitor="eval/psnr",
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
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts"))
    parser.add_argument("--yolo_path", type=str, default="")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)