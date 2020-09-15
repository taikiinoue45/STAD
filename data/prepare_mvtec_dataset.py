import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def create_info_csv() -> pd.DataFrame:

    base = Path("/app/STAD/data")
    di: dict = {
        "old_image_path": [],
        "old_stem": [],
        "defect_type": [],
        "data_type": [],
        "category": [],
    }
    for p in base.glob("*/*/*/*.png"):
        di["old_image_path"].append(str(p))
        di["old_stem"].append(p.stem)
        di["defect_type"].append(p.parents[0].name)
        di["data_type"].append(p.parents[1].name)
        di["category"].append(p.parents[2].name)

    df = pd.DataFrame(di)
    df = df.loc[df["data_type"] != "ground_truth", :]
    df["stem"] = -1
    df["old_mask_path"] = -1
    for i in df.index:
        old_stem = df.loc[i, "old_stem"]
        defect_type = df.loc[i, "defect_type"]
        data_type = df.loc[i, "data_type"]
        category = df.loc[i, "category"]

        stem = f"{category}_{data_type}_{defect_type}_{old_stem}"
        old_mask_path = f"/app/STAD/data/{category}/ground_truth/{defect_type}/{old_stem}_mask.png"
        df.loc[i, "stem"] = stem
        df.loc[i, "old_mask_path"] = old_mask_path

    df.to_csv("/app/STAD/data/info.csv", index=False)
    return df


def move_images_and_masks(df: pd.DataFrame):

    os.mkdir("/app/STAD/data/images")
    os.mkdir("/app/STAD/data/masks")

    for i in df.index:
        old_image_path = df.loc[i, "old_image_path"]
        old_mask_path = df.loc[i, "old_mask_path"]
        stem = df.loc[i, "stem"]

        if os.path.exists(old_mask_path):
            os.rename(old_mask_path, f"/app/STAD/data/masks/{stem}.png")
        else:
            img = cv2.imread(old_image_path)
            mask = np.zeros(img.shape)
            cv2.imwrite(f"/app/STAD/data/masks/{stem}.png", mask)

        os.rename(old_image_path, f"/app/STAD/data/images/{stem}.png")


if __name__ == "__main__":

    df = create_info_csv()
    move_images_and_masks(df)
