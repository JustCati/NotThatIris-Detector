
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

    out_path = os.path.join(os.path.dirname(csv_path), "feature_extractor", "normalized_iris.csv")
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



def split_iris_thousand_users(csv_path,
                             known_user_ratio=0.8,
                             unknown_user_ratio=0.1):
    known_test_ratio = 0.1
    known_train_ratio = 0.8
    unknown_test_ratio = 0.5

    random.seed(4242)
    df = pd.read_csv(csv_path, index_col=0)
    users = df["Label"].apply(lambda x: x.split("-")[0]).unique()

    random.shuffle(users)
    known_users = users[:int(len(users)*known_user_ratio)]
    unknown_users = users[int(len(users)*known_user_ratio):int(len(users)*(known_user_ratio+unknown_user_ratio))]
    negative_users = users[int(len(users)*(known_user_ratio+unknown_user_ratio)):]

    unknown_users_test = unknown_users[:int(len(unknown_users)*unknown_test_ratio)]
    unknown_users_val = unknown_users[int(len(unknown_users)*unknown_test_ratio):]

    negativeDF = df[df["Label"].apply(lambda x: x.split("-")[0] in negative_users)]
    negativeDF["Label"] = -1

    unknown_testDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_test)]
    unknown_valDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_val)]

    knownDF = df[df["Label"].apply(lambda x: x.split("-")[0] in known_users)]
    known_train = knownDF.sample(frac=known_train_ratio, random_state=4242)
    knownDF = knownDF.drop(known_train.index)
    known_test = knownDF.sample(frac=known_test_ratio, random_state=4242)
    knownDF = knownDF.drop(known_test.index)
    known_val = knownDF

    assert set(unknown_testDF.Label.unique()).intersection(set(known_train.Label.unique())) == set()
    assert set(unknown_testDF.Label.unique()).intersection(set(known_test.Label.unique())) == set()
    assert set(unknown_testDF.Label.unique()).intersection(set(known_val.Label.unique())) == set()

    assert set(unknown_testDF.Label.unique()).intersection(set(unknown_valDF.Label.unique())) == set()

    assert set(negativeDF.Label.unique()).intersection(set(known_train.Label.unique())) == set()
    assert set(negativeDF.Label.unique()).intersection(set(known_test.Label.unique())) == set()
    assert set(negativeDF.Label.unique()).intersection(set(known_val.Label.unique())) == set()
    assert set(negativeDF.Label.unique()).intersection(set(unknown_testDF.Label.unique())) == set()
    assert set(negativeDF.Label.unique()).intersection(set(unknown_valDF.Label.unique())) == set()

    train_df = pd.concat([known_train, negativeDF])
    test_df = pd.concat([known_test, unknown_testDF])
    val_df = pd.concat([known_val, unknown_valDF])

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(csv_path.replace(os.path.basename(csv_path), "train_users.csv"))
    test_df.to_csv(csv_path.replace(os.path.basename(csv_path), "test_users.csv"))
    val_df.to_csv(csv_path.replace(os.path.basename(csv_path), "val_users.csv"))
