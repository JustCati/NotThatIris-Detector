import os
import torch 
import argparse
from torch.utils.data import DataLoader

import lightning as L
from torchvision.transforms import v2 as T
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.models.resnet import Resnet
from src.dataset.iris_thousand import IrisThousand
from src.utils.dataset_utils.iris_thousand import normalize_iris_thousand, split_iris_thousand



def main(args):
    dataset_path = args.dataset_path
    model_name = args.model_name
    root_dir = args.output_path
    root_dir = os.path.join(root_dir, model_name.upper())

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    images_path = os.path.join(dataset_path, "images")
    test_csv_path = os.path.join(dataset_path, "test_iris.csv")
    train_csv_path = os.path.join(dataset_path, "train_iris.csv")
    complete_csv_path = os.path.join(dataset_path, "iris_thousands.csv")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        if not os.path.exists(os.path.join(dataset_path, "normalized_iris.csv")):
            out_path = normalize_iris_thousand(images_path, os.path.join(dataset_path, "iris_thousands.csv"))
        else:
            out_path = os.path.join(dataset_path, "normalized_iris.csv")
            split_iris_thousand(out_path)

    transform = T.Compose([
        T.GaussianBlur(kernel_size=3),
        T.JPEG(quality=(50, 75)),
    ])

    train_dataset = IrisThousand(train_csv_path, images_path, complete_csv_path, transform=transform)
    eval_dataset = IrisThousand(test_csv_path, images_path, complete_csv_path, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = Resnet(model_name, num_classes=train_dataset.num_classes, batch_size=args.batch_size)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="iris-thousand")
    tb_logger = TensorBoardLogger(os.path.join(root_dir, "logs"), name="iris-thousand", version=csv_logger.version)

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
        accelerator="gpu",
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
    parser.add_argument("--model_name", type=str, default="RESNET18", choices=["RESNET18", "RESNET50", "resnet18", "resnet50"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
