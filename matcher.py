import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import multiprocessing
from torch.utils.data import DataLoader

from src.models.resnet import FeatureExtractor
from src.models.mlp_matcher import MLPMatcher
from src.models.gallery_classifier import Matcher
from src.graphics.metrics import roc_graph, far_frr_graph
from src.dataset.GenericIrisDataset import GenericIrisDataset
from src.models.GenericFeatureExtractor import GenericFeatureExtractor
from src.engine.thresholding import evaluate_vectorbased, get_eer, evaluate_mlp
from src.utils.dataset_utils.upsample import generate_upsampled_normalized_iris
from src.utils.dataset_utils.iris import normalize_iris_thousand, split_iris_thousand_users



def get_label_map(csv_file):
    df = pd.read_csv(csv_file, index_col=0)
    label_map = {label: i for i, label in enumerate(df["Label"].unique())}
    label_map.update({"-1": -1})
    return label_map




def main(args):
    sr_model_path = args.sr_model_path
    resnet_model_path = args.resnet_model_path
    dataset_path = args.dataset_path
    persistent_outpath = args.out_path
    collection_name = args.collection_name
    device = "cuda" if torch.cuda.is_available() else "cpu"

    images_path = os.path.join(dataset_path, "images")
    test_csv_path = os.path.join(dataset_path, "test_users.csv")
    train_csv_path = os.path.join(dataset_path, "train_users.csv")
    eval_csv_path = os.path.join(dataset_path, "val_users.csv")
    complete_csv_path = os.path.join(dataset_path, "iris_thousands.csv")


    #* -------- CHECK IF THE DATASET IS PREPROCESSED -----------
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path) or not os.path.exists(eval_csv_path):
        if not os.path.exists(os.path.join(dataset_path, "normalized_iris.csv")):
            csv_normalized_path = normalize_iris_thousand(images_path, complete_csv_path)
        else:
            csv_normalized_path = os.path.join(dataset_path, "normalized_iris.csv")
        split_iris_thousand_users(csv_normalized_path)


    #* ------------- LOAD DATASET -------------
    label_map = get_label_map(train_csv_path)
    train_dataset = GenericIrisDataset(train_csv_path, 
                                       images_path,
                                       complete_csv_path,
                                       label_map=label_map,
                                       modality="user",
                                       keep_uknown=False,
                                       upsample=args.upsample)
    test_dataset = GenericIrisDataset(test_csv_path,
                                      images_path,
                                      complete_csv_path,
                                      label_map=label_map,
                                      keep_uknown=True,
                                      upsample=args.upsample)
    eval_dataset = GenericIrisDataset(eval_csv_path,
                                      images_path,
                                      complete_csv_path,
                                      label_map=label_map,
                                      keep_uknown=True,
                                      upsample=args.upsample)

    cpu_count = multiprocessing.cpu_count() // 2
    batch_size = 1 if args.type == "vector" else 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=cpu_count)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count)


    #* ------------- LOAD FEATURE EXTRACTOR -------------
    feat_extractor = FeatureExtractor(model_path=resnet_model_path, num_classes=819).to(device)


    #* ------------- LOAD MATCHER -------------
    if args.type == "vector":
        matcher = Matcher(model=feat_extractor, 
                        collection_name=collection_name,
                        out_path=persistent_outpath,
                        device=device)
    elif args.type == "mlp":
        mlp_model_path = args.mlp_model_path
        matcher = MLPMatcher.load_from_checkpoint(mlp_model_path,
                                                  in_feature=2048,
                                                  num_classes=1597,
                                                  extractor=feat_extractor).to(device)
        matcher.eval()


    #* --------- LOAD USERS INTO MATCHER (ONLY FOR VECTOR-BASED MATCHER) ---------
    if args.type == "vector":
        print("Loading users into matcher...")
        for imgs, labels in tqdm(train_dataloader):
            matcher.add_user(imgs, labels)
        print()


    #* ---------- FINE TUNE THE THRESHOLD OF THE MATCHER ----------
    print("Fine tuning the threshold of the matcher...")
    if args.type == "vector":
        y, y_pred = evaluate_vectorbased(matcher, test_dataloader, train=True)
    else:
        y, y_pred = evaluate_mlp(matcher, test_dataloader)
    far, frr, tpr, threshold, eer_index, eer_threshold = get_eer(y, y_pred)
    print(f"FAR: {far[eer_index]:.4f}, FRR: {frr[eer_index]:.4f}, Threshold at EER: {eer_threshold:.4f}")
    print()



    #* ----------- EVALUATE MATCHER -----------
    print("Evaluating the matcher...")
    matcher.set_threshold(eer_threshold)
    if args.type == "vector":
        y, y_pred = evaluate_vectorbased(matcher, eval_dataloader)
    else:
        y, y_pred = evaluate_mlp(matcher, eval_dataloader)
    print()

    eval_far, eval_frr, _, _, _, _ = get_eer(y, y_pred)
    print(f"FAR: {eval_far[eer_index]:.4f}, FRR: {eval_frr[eer_index]:.4f}, Threshold Used: {eer_threshold:.4f}")
    print()


    if args.plot:
        roc_graph(far, tpr, y, y_pred)
        far_frr_graph(far, frr, threshold, eer_index, eval_far, eval_frr)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matcher")
    parser.add_argument("type", type=str, help="Type of the matcher", choices=["vector", "mlp"])
    parser.add_argument("--resnet_model_path", type=str, required=True, help="Path to the feature extraction model")
    parser.add_argument("--mlp_model_path", type=str, default=None, help="Path to the MLP model")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the graphs")
    parser.add_argument("--upsample", action="store_true", help="Whether to use Super Resolution on the images")
    parser.add_argument("--collection_name", type=str, help="Name of the collection", default="Iris-Matcher")
    parser.add_argument("--out_path", type=str, help="Path to the output directory", default=os.path.join(os.path.dirname(__file__), "Vector-Based-Matcher"))
    parser.add_argument("--dataset_path", type=str, default=os.path.join(os.path.dirname(__file__), "datasets", "Iris-Thousand"))
    args = parser.parse_args()
    main(args)
