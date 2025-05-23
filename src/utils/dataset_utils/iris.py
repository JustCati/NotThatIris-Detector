import os
import cv2
import pandas as pd
from src.utils.eyes import normalize_eye



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



def build_df(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
    if dataset_path.endswith("/"):
        dataset_path = dataset_path[:-1]
    if not os.path.basename(dataset_path) == "images":
        img_path = os.path.join(dataset_path, "images")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Dataset image path {img_path} does not exist.")

    data = []
    for user in os.listdir(img_path):
        user_path = os.path.join(img_path, user)
        for side in sorted(os.listdir(user_path)):
            side_path = os.path.join(user_path, side)
            for img in os.listdir(side_path):
                if not img.endswith(".jpg"):
                    continue
                user_id = f"{img[2:5]}-{img[5]}"
                data.append(
                    {
                        "Label": user_id,
                        "ImagePath": os.path.join(side_path, img),
                    }
                )
    df = pd.DataFrame(data)
    return df


def split_by_sample(dataset_path, train_ration=0.8):
    df = build_df(dataset_path)
    df = df.sample(frac=1, random_state=4242).reset_index(drop=True)

    train_df = df.iloc[:int(len(df)*train_ration)]
    test_df = df.iloc[int(len(df)*train_ration):]

    assert len(train_df) + len(test_df) == len(df)
    assert train_df.index.intersection(test_df.index).empty

    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(os.path.join(dataset_path, "train_iris.csv"))
    test_df.to_csv(os.path.join(dataset_path, "test_iris.csv"))







# def split_iris_thousand_users(csv_path,
#                               known_ratio=0.8):
#     random.seed(4242)
#     df = pd.read_csv(csv_path, index_col=0)
#     users = df["Label"].apply(lambda x: x.split("-")[0]).unique()

#     random.shuffle(users)
#     known_users = users[:int(len(users)*known_ratio)]
#     unknown_users = users[int(len(users)*known_ratio):]

#     train_df = df[df["Label"].apply(lambda x: x.split("-")[0] in known_users)]

#     unknown_users_train = unknown_users[:int(len(unknown_users)*0.6)]
#     unknown_users_test = unknown_users[int(len(unknown_users)*0.6):int(len(unknown_users)*0.8)]
#     unknown_users_val = unknown_users[int(len(unknown_users)*0.8):]

#     unknown_trainDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_train)]
#     unknown_testDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_test)]
#     unknown_valDF = df[df["Label"].apply(lambda x: x.split("-")[0] in unknown_users_val)]

#     known_trainDF = train_df.sample(frac=0.8, random_state=4242)
#     known_testDF = train_df.drop(known_trainDF.index).sample(frac=0.1, random_state=4242)
#     known_valDF = train_df.drop(known_trainDF.index).drop(known_testDF.index)

#     assert set(unknown_testDF.Label.unique()).intersection(set(known_trainDF.Label.unique())) == set()
#     assert set(unknown_testDF.Label.unique()).intersection(set(known_testDF.Label.unique())) == set()
#     assert set(unknown_testDF.Label.unique()).intersection(set(known_valDF.Label.unique())) == set()

#     assert set(unknown_testDF.Label.unique()).intersection(set(unknown_valDF.Label.unique())) == set()

#     unknown_testDF["Label"] = -1
#     unknown_valDF["Label"] = -1
#     unknown_trainDF["Label"] = -1

#     train_df = pd.concat([known_trainDF, unknown_trainDF])
#     test_df = pd.concat([known_testDF, unknown_testDF])
#     val_df = pd.concat([known_valDF, unknown_valDF])

#     train_df.reset_index(drop=True, inplace=True)
#     test_df.reset_index(drop=True, inplace=True)
#     val_df.reset_index(drop=True, inplace=True)

#     train_df.to_csv(csv_path.replace(os.path.basename(csv_path), "train_users.csv"))
#     test_df.to_csv(csv_path.replace(os.path.basename(csv_path), "test_users.csv"))
#     val_df.to_csv(csv_path.replace(os.path.basename(csv_path), "val_users.csv"))
