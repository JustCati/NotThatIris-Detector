import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.utils.eyes import normalize_eye



def upsample(img, model, window_size=16, device='cpu', SCALE=4):
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

        output = model(img)
        output = output[..., :h_old * SCALE, :w_old * SCALE]
        output = output.data.squeeze().float().permute(1, 2, 0).cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        return Image.fromarray(output)



def generate_upsampled_normalized_iris(model, csv_file, low_res_path, device="cpu"):
    imgs = os.listdir(low_res_path)
    map_lq = {f"{img.split('_')[0]}.jpg": img for img in imgs}

    df = pd.read_csv(csv_file, index_col=0)
    for i in tqdm(range(len(df))):
        original_path = df.iloc[i]["ImagePath"]
        img_path = os.path.join(low_res_path, map_lq[os.path.basename(original_path)])

        img = Image.open(img_path)
        img = upsample(img, model, device=device)
        img = np.array(img)
        img = normalize_eye(img, radius_out=60, find_iris=False)
        img = Image.fromarray(img)

        out_path = os.path.join(original_path.replace("normalized_iris", "upsampled_iris"))
        assert os.path.basename(out_path) == os.path.basename(img_path).split("_")[0] + ".jpg"

        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        img.save(out_path)
