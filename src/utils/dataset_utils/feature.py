import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torchvision.transforms import v2 as T



def extract_feature_from_normalized_iris(model, csv_file, device="cpu"):
    df = pd.read_csv(csv_file, index_col=0)
    for i in tqdm(range(len(df))):
        img_path = df.iloc[i]["ImagePath"]
        print(img_path)

        img = Image.open(img_path)
        img = T.ToTensor()(img)
        img = img.unsqueeze(0)
        feature_vector = model(img.to(device)).cpu().numpy()[0]
        print(feature_vector.shape)

        out_path = img_path.replace("normalized_iris", "feature_iris")
        out_path = out_path.replace(".jpg", "")
        print(out_path)

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        np.save(out_path, feature_vector)
