
import os
import cv2
import pandas as pd
from src.utils.eyes import normalize_eye



def normalize_iris_thousand(dataset_path, csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    df["ImagePath"] = df["ImagePath"].apply(
                lambda x: x.replace(
                    "/kaggle/input/casia-iris-thousand/CASIA-Iris-Thousand",
                    dataset_path
                )
            )
    df["outputPath"] = df["ImagePath"].apply(
                lambda x: x.replace("images", "normalized_iris")
            )

    for id, row in df.iterrows():
        input = row["ImagePath"]
        output = row["outputPath"]

        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))

        try:
            norm = normalize_eye(input)
            cv2.imwrite(output, norm)
        except Exception as e:
            print(f"Error: {e}")
            df.drop(id, inplace=True)
            continue

    df.drop(columns=["ImagePath"], inplace=True)
    df.rename(columns={"outputPath": "ImagePath"}, inplace=True)

    out_path = csv_path.replace(os.path.basename(csv_path), "normalized_iris.csv")
    df.to_csv(out_path)
    return out_path



def split_iris_thousand(csv_path, train_ration=0.8):
    df = pd.read_csv(csv_path, index_col=0)
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    train_df = df.iloc[:int(len(df)*train_ration)]
    test_df = df.iloc[int(len(df)*train_ration):]

    assert len(train_df) + len(test_df) == len(df)
    assert train_df.index.intersection(test_df.index).empty

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(csv_path.replace(os.path.basename(csv_path), "train_iris.csv"))
    test_df.to_csv(csv_path.replace(os.path.basename(csv_path), "test_iris.csv"))
