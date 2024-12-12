import os
import torch 
import argparse
from torch.utils.data import DataLoader

import lightning as L
from pytorch_lightning.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


from src.models.resnet import Resnet
from src.dataset.iris_thousand import IrisThousand




def main(args):
    root_dir = args.output_path
    dataset_path = args.dataset_path

    images_path = os.path.join(dataset_path, "images")
    train_csv_path = os.path.join(dataset_path, "train_iris.csv")
    test_csv_path = os.path.join(dataset_path, "test_iris.csv")

    train_dataset = IrisThousand(train_csv_path, images_path)
    eval_dataset = IrisThousand(test_csv_path, images_path)

    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=512, shuffle=False)

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    model = Resnet(args.batch_size, args.num_classes)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="iris-thousand")


    best_checkpoint_saver = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "models"),
        filename="best",
        save_top_k=1,
        monitor="f1",
        mode="max",
        verbose=True,
        save_last=True
        )

    early_stop_callback = EarlyStopping(
        monitor="f1",
        min_delta=0.05,
        patience=100,
        verbose=False, 
        mode="max"
        )

    trainer = L.Trainer(
        default_root_dir=root_dir,   
        max_epochs=args.num_epochs,
        accelerator="gpu",
        logger=csv_logger,
        callbacks=[best_checkpoint_saver]#, early_stop_callback]
        )

    trainer.fit(model=model, 
                train_dataloaders=train_dataloader,
                val_dataloaders=eval_dataloader
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts", "RESNET50"))
    parser.add_argument("--num_epochs", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=2000)
    args = parser.parse_args()
    main(args)
