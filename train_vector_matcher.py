import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models.gallery_classifier import Matcher
from src.engine.thresholding import evaluate, get_eer
from src.graphics.metrics import roc_graph, far_frr_graph
from src.dataset.GenericIrisDataset import GenericIrisDataset
from src.utils.dataset_utils.iris import normalize_iris_thousand, split_iris_thousand_users




def main(args):
    model_path = args.model_path
    dataset_path = args.dataset_path
    persistent_outpath = args.out_path
    collection_name = args.collection_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_path = os.path.join(dataset_path, "images")
    test_csv_path = os.path.join(dataset_path, "test_users.csv")
    train_csv_path = os.path.join(dataset_path, "train_users.csv")
    eval_csv_path = os.path.join(dataset_path, "val_users.csv")
    complete_csv_path = os.path.join(dataset_path, "iris_thousands.csv")

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path) or not os.path.exists(eval_csv_path):
        if not os.path.exists(os.path.join(dataset_path, "normalized_iris.csv")):
            out_path = normalize_iris_thousand(images_path, complete_csv_path)
        else:
            out_path = os.path.join(dataset_path, "normalized_iris.csv")
        split_iris_thousand_users(out_path)


    train_dataset = GenericIrisDataset(train_csv_path, images_path, complete_csv_path, modality="user")
    test_dataset = GenericIrisDataset(test_csv_path, images_path, complete_csv_path)
    eval_dataset = GenericIrisDataset(eval_csv_path, images_path, complete_csv_path)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    matcher = Matcher(model_path=model_path, 
                      collection_name=collection_name,
                      out_path=persistent_outpath,
                      device=device)


    #* LOAD USERS INTO MATCHER
    print("Loading users into matcher...")
    for imgs, labels in tqdm(train_dataloader):
        matcher.add_user(imgs, labels)
    print()


    #* FINE TUNE THE THRESHOLD OF THE MATCHER
    print("Fine tuning the threshold of the matcher...")
    y, y_pred = evaluate(matcher, test_dataloader, train=True)
    far, frr, tpr, threshold, eer_index, eer_threshold = get_eer(y, y_pred)
    print(f"FAR: {far[eer_index]:.4f}, FRR: {frr[eer_index]:.4f}, Threshold at EER: {eer_threshold:.4f}")
    print()

    if args.graphs:
        roc_graph(far, tpr, y, y_pred)
        far_frr_graph(far, frr, threshold, eer_index)


    #* EVALUATE MATCHER
    print("Evaluating the matcher...")
    matcher.set_threshold(eer_threshold) #? Set the threshold to the EER threshold
    y, y_pred = evaluate(matcher, eval_dataloader)
    print()

    far, frr, _, _, eer_index, eer = get_eer(y, y_pred)
    print(f"FAR: {far[eer_index]:.4f}, FRR: {frr[eer_index]:.4f}, Threshold at EER: {eer:.4f}")
    print()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matcher")
    parser.add_argument("--model_path", type=str, help="Path to the feature extraction model")
    parser.add_argument("--graphs", action="store_true", help="Whether to plot the graphs")
    parser.add_argument("--collection_name", type=str, help="Name of the collection", default="Iris-Matcher")
    parser.add_argument("--out_path", type=str, help="Path to the output directory", default=os.path.join(os.path.dirname(__file__), "Iris-Matcher"))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    args = parser.parse_args()
    main(args)
