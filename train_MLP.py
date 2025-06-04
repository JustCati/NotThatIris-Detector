import os
import torch 
import argparse
import pandas as pd
import multiprocessing
from torch.utils.data import DataLoader
from src.engine.thresholding import get_eer, evaluate_mlp

import lightning as L
from torchvision.transforms import v2 as T
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from src.models.yolo import getYOLO
from src.models.mlp_matcher import MLPMatcher
from src.models.backbone import FeatureExtractor
from src.dataset.NormalizedDataset import NormalizedIrisDataset
from src.models.GenericFeatureExtractor import GenericFeatureExtractor
from src.utils.dataset_utils.iris import normalize_dataset, split_by_user



def get_label_map(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    label_map = {label: i for i, label in enumerate(df["Label"].unique())}
    label_map.update({"-1": -1})
    return label_map


def main(args):
    root_dir = args.output_path
    dataset_path = args.dataset_path
    root_dir = os.path.join(root_dir, "MLPMATCHER")

    L.seed_everything(4242, workers=True)
    torch.set_float32_matmul_precision("high")

    images_path = os.path.join(dataset_path, "images_raw")
    test_csv_path = os.path.join(dataset_path, "test_users.csv")
    train_csv_path = os.path.join(dataset_path, "train_users.csv")
    val_csv_path = os.path.join(dataset_path, "val_users.csv")
    normalized_path = os.path.join(dataset_path, "normalized")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        if not os.path.exists(normalized_path):
            print("Normalizing dataset...")
            yolo_det = getYOLO(
                checkpoint_path=args.yolo_det_path,
                task="detection",
                device="cuda" if torch.cuda.is_available() else "cpu",
                inference=True
            )
            yolo_seg = getYOLO(
                checkpoint_path=args.yolo_seg_path,
                task="segment",
                device="cuda" if torch.cuda.is_available() else "cpu",
                inference=True
            )
            normalize_dataset((yolo_det, yolo_seg), dataset_path, distance=True)
        split_by_user(dataset_path)

    transform = T.Compose([
        T.GaussianBlur(kernel_size=3),
        T.JPEG(quality=(50, 75)),
    ])


    train_df = pd.read_csv(train_csv_path, index_col=0)
    test_df = pd.read_csv(test_csv_path, index_col=0)
    val_df = pd.read_csv(val_csv_path, index_col=0)
    
    train_df = train_df[train_df["Label"] != -1]
    test_df = test_df[test_df["Label"] != -1]
    val_df = val_df[val_df["Label"] != -1]
    
    train_users = train_df["Label"].unique()
    test_users = test_df["Label"].unique()
    val_users = val_df["Label"].unique()
    all_users = list(set(train_users) | set(test_users) | set(val_users))
    
    train_dataset = NormalizedIrisDataset(
        csv_file=train_csv_path,
        transform=transform,
        classes=all_users
    )
    test_dataset = NormalizedIrisDataset(
        csv_file=test_csv_path,
        transform=transform,
        classes=all_users,
        keep_unknown=True
    )
    val_dataset = NormalizedIrisDataset(
        csv_file=val_csv_path,
        transform=transform,
        classes=all_users,
        keep_unknown=True
    )
    
    mapper = train_dataset.get_mapper()
    with open("label_map.txt", "w") as f:
        for k, v in mapper.items():
            f.write(f"{k}: {v}\n")

    cpu_count = multiprocessing.cpu_count() // 2
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=cpu_count)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=cpu_count)

    num_classes = len(train_dataset.get_active_labels())
    print(f"Number of classes: {num_classes}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = FeatureExtractor(model_path=args.backbone_path).to(device)
    vector_dim = backbone.get_vector_dim()
    
    model = MLPMatcher(in_feature=vector_dim, num_classes=num_classes, extractor=backbone, verbose=True).to(device)
    csv_logger = CSVLogger(os.path.join(root_dir, "logs"), name="mlp")
    tb_logger = TensorBoardLogger(os.path.join(root_dir, "logs"), name="mlp", version=csv_logger.version)

    if not args.threshold:
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

    print("Training completed.")
    print("Finding best threshold using the test set...")
    
    best_test_threshold = 0
    model = MLPMatcher.load_from_checkpoint(os.path.join(root_dir, "models", "best.ckpt"), 
                                            in_feature=vector_dim,
                                            num_classes=num_classes)
    model.set_extractor(backbone)
    model.to(device)
    model.eval()
    
    gt, pred = evaluate_mlp(model, test_dataloader)
    _, frr, _, _, eer_index, best_test_threshold = get_eer(gt, pred)
    print(f"Best threshold found: {best_test_threshold:2f} with EER: {frr[eer_index]:2f} at index {eer_index:2f}")
    model.set_threshold(best_test_threshold)
    
    print("Evaluating on the test set...")
    gt, pred = evaluate_mlp(model, val_dataloader)
    _, frr, _, _, val_eer_index, eer_threshold = get_eer(gt, pred)
    print(f"Best Eval EER: {frr[val_eer_index]:2f} at threshold {eer_threshold:2f} (index {val_eer_index:2f})")
    print(f"Real EER on validation set: {frr[eer_index]:2f} at threshold {best_test_threshold:2f}")

    model_checkpoint = torch.load(os.path.join(root_dir, "models", "best.ckpt"))
    model_checkpoint["threshold"] = best_test_threshold
    torch.save(model_checkpoint, os.path.join(root_dir, "models", "best_with_threshold.ckpt"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Distance"))
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to the feature extraction model")
    parser.add_argument("--yolo_det_path", type=str, required=False, help="Path to the eye detector checkpoint")
    parser.add_argument("--yolo_seg_path", type=str, required=False, help="Path to the iris segmentation checkpoint")
    parser.add_argument("--output_path", type=str, default=os.path.join(os.path.dirname(__file__), "ckpts"))
    parser.add_argument("--upsample", action="store_true", default=False, help="Use upsampled dataset")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--threshold", action="store_true", default=False, help="Use thresholding for evaluation")
    args = parser.parse_args()
    main(args)
