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

from src.models.mlp_matcher import MLPMatcher
from src.models.adapter import FeatureAdapter
from src.models.resnet import FeatureExtractor
from src.dataset.GenericIrisDataset import GenericIrisDataset
from src.models.GenericFeatureExtractor import GenericFeatureExtractor
from src.utils.dataset_utils.iris import normalize_iris_lamp, split_iris_lamp
from src.utils.dataset_utils.feature import extract_feature_from_normalized_iris



def get_label_map(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    label_map = {label: i for i, label in enumerate(df["Label"].unique())}
    label_map.update({"-1": -1})
    return label_map


def main(args):
    root_dir = args.output_path
    dataset_path = args.dataset_path
    if args.adapter_model_path is not None:
        root_dir = os.path.join(root_dir, "MLPMATCHER_ADAPTER")
    else:
        root_dir = os.path.join(root_dir, "MLPMATCHER")

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    images_path = os.path.join(dataset_path, "images")
    test_csv_path = os.path.join(dataset_path, "test_users.csv")
    train_csv_path = os.path.join(dataset_path, "train_users.csv")


    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        if not os.path.exists(os.path.join(dataset_path, "normalized_iris.csv")):
            out_path = normalize_iris_lamp(images_path)
        else:
            out_path = os.path.join(dataset_path, "normalized_iris.csv")
        split_iris_lamp(out_path)
    complete_csv_path = os.path.join(dataset_path, "normalized_iris.csv")

    if not os.path.exists(os.path.join(dataset_path, "feature_iris")):
        feat_model_path = args.feature_model_path
        feat_model = FeatureExtractor(model_path=feat_model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        feat_model.to(device)
        extract_feature_from_normalized_iris(feat_model, os.path.join(dataset_path, "normalized_iris.csv"), device=device)

    transform = T.Compose([
        T.GaussianBlur(kernel_size=3),
        T.JPEG(quality=(50, 75)),
    ])

    label_map = get_label_map(train_csv_path)
    train_dataset = GenericIrisDataset(train_csv_path, 
                                       images_path,
                                       complete_csv_path,
                                       label_map=label_map,
                                       keep_uknown=False,
                                       upsample=args.upsample,
                                       transform=transform)
    test_dataset = GenericIrisDataset(test_csv_path, 
                                      images_path, 
                                      complete_csv_path,
                                      label_map=label_map,
                                      keep_uknown=True,
                                      upsample=args.upsample,
                                      transform=transform)

    cpu_count = multiprocessing.cpu_count() // 2
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count)


    VECTOR_DIM = 2048
    num_classes = len(train_dataset.get_active_labels())
    feat_model_path = args.feature_model_path
    device = "cuda" if torch.cuda.is_available() else "cpu"


    extractor = FeatureExtractor(model_path=feat_model_path, num_classes=819).to(device)
    if args.adapter_model_path:
        if os.path.exists(args.adapter_model_path):
            adapter_model_path = args.adapter_model_path
            adapter_model = FeatureAdapter(model_path=adapter_model_path, num_classes=2048).to(device)
            extractor = GenericFeatureExtractor(modules=[extractor, adapter_model]).to(device)
        else:
            print("Adapter model not found, using the feature extractor only...")


    model = MLPMatcher(in_feature=VECTOR_DIM, num_classes=num_classes, extractor=extractor, verbose=True).to(device)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="iris-thousand")
    tb_logger = TensorBoardLogger(os.path.join(root_dir, "logs"), name="iris-thousand", version=csv_logger.version)

    best_checkpoint_saver = ModelCheckpoint(
        dirpath=os.path.join(root_dir, "models"),
        filename="best",
        save_top_k=1,
        monitor="eval/acc+eer",
        mode="max",
        verbose=True,
        save_last=True
        )

    early_stop_callback = EarlyStopping(
        monitor="eval/acc+eer",
        min_delta=0.004,
        patience=10,
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
                val_dataloaders=test_dataloader
                )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    parser.add_argument("--feature_model_path", type=str, required=True, help="Path to the feature extraction model")
    parser.add_argument("--adapter_model_path", type=str, required=False, help="Path to the adapter model")
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts"))
    parser.add_argument("--upsample", action="store_true", default=False, help="Use upsampled dataset")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)
