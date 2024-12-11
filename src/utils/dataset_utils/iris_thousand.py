
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
    df.to_csv(csv_path)


if __name__ == "__main__":
    dataset_path = os.path.join(os.getcwd(), "datasets", "Iris-Thousand", "images")
    csv_path = os.path.join(os.getcwd(), "datasets", "Iris-Thousand", "iris_thousands.csv")
    normalize_iris_thousand(dataset_path, csv_path)
