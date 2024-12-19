
import os
import cv2
import random
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

    out_path = os.path.join(os.path.dirname(csv_path), "normalized_iris.csv")
    df.to_csv(out_path)
    return out_path



def normalize_iris_lamp(dataset_path):
    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg"):
                id = root.split("/")[-2] + "-" + root.split("/")[-1]
                input = os.path.join(root, file)
                output = input.replace("images", "normalized_iris")

                if not os.path.exists(os.path.dirname(output)):
                    os.makedirs(os.path.dirname(output))

                try:
                    norm = normalize_eye(input)
                    cv2.imwrite(output, norm)
                    info = (id, output)
                    data.append(info)
                except Exception as e:
                    print(f"Error: {e}")
                    continue

    df = pd.DataFrame(data, columns=["Label", "ImagePath"])
    out_path = os.path.join(os.path.dirname(dataset_path), "normalized_iris.csv")
    df.to_csv(out_path)
    return out_path



def split_iris_lamp(csv_path, train_ration=0.8):
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



def split_iris_thousand_users(csv_path,
                             known_ratio=0.8):
    random.seed(4242)
    df = pd.read_csv(csv_path, index_col=0)
    users = df["Label"].apply(lambda x: x.split("-")[0]).unique()

    random.shuffle(users)
    known_users = users[:int(len(users)*known_ratio)]
    unknown_users = users[int(len(users)*known_ratio):]

    train_df = df[df["Label"].apply(lambda x: x.split("-")[0] in known_users)]

    unknown_users_test = unknown_users[:int(len(unknown_users)*0.5)]
    unknown_users_val = unknown_users[int(len(unknown_users)*0.5):]

    unknown_testDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_test)]
    unknown_valDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_val)]

    known_trainDF = train_df.sample(frac=0.8, random_state=4242)
    known_testDF = train_df.drop(known_trainDF.index).sample(frac=0.1, random_state=4242)
    known_valDF = train_df.drop(known_trainDF.index).drop(known_testDF.index)

    assert set(unknown_testDF.Label.unique()).intersection(set(known_trainDF.Label.unique())) == set()
    assert set(unknown_testDF.Label.unique()).intersection(set(known_testDF.Label.unique())) == set()
    assert set(unknown_testDF.Label.unique()).intersection(set(known_valDF.Label.unique())) == set()

    assert set(unknown_testDF.Label.unique()).intersection(set(unknown_valDF.Label.unique())) == set()

    unknown_testDF["Label"] = -1
    unknown_valDF["Label"] = -1

    train_df = pd.concat([known_trainDF])
    test_df = pd.concat([known_testDF, unknown_testDF])
    val_df = pd.concat([known_valDF, unknown_valDF])

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(csv_path.replace(os.path.basename(csv_path), "train_users.csv"))
    test_df.to_csv(csv_path.replace(os.path.basename(csv_path), "test_users.csv"))
    val_df.to_csv(csv_path.replace(os.path.basename(csv_path), "val_users.csv"))
