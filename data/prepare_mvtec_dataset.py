import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def create_info_csv() -> DataFrame:

    mvtec_dir = Path("/dgx/github/STAD/data/MVTec")
    di: Dict[str, List[str]] = {
        "old_img_path": [],
        "old_stem": [],
        "defect": [],
        "mode": [],
        "category": [],
    }

    for p in mvtec_dir.glob("*/[!ground]*/*/*.png"):
        di["old_img_path"].append(str(p))
        di["old_stem"].append(p.stem)
        di["defect"].append(p.parents[0].name)
        di["mode"].append(p.parents[1].name)  # train or test
        di["category"].append(p.parents[2].name)

    df = pd.DataFrame(di)
    df["stem"] = ""
    df["old_mask_path"] = ""
    for i in df.index:
        old_stem = df.loc[i, "old_stem"]
        defect = df.loc[i, "defect"]
        mode = df.loc[i, "mode"]
        category = df.loc[i, "category"]

        stem = f"{category}_{mode}_{defect}_{old_stem}"
        old_mask_path = str(mvtec_dir / f"{category}/ground_truth/{defect}/{old_stem}_mask.png")
        df.loc[i, "stem"] = stem
        df.loc[i, "old_mask_path"] = old_mask_path

    df.to_csv("/dgx/github/STAD/data/info.csv", index=False)
    return df


def move_images_and_masks(df: pd.DataFrame) -> None:

    os.mkdir("/dgx/github/STAD/data/images")
    os.mkdir("/dgx/github/STAD/data/masks")

    for i in tqdm(df.index):
        old_img_path = df.loc[i, "old_img_path"]
        old_mask_path = df.loc[i, "old_mask_path"]
        stem = df.loc[i, "stem"]

        if os.path.exists(old_mask_path):
            os.rename(old_mask_path, f"/dgx/github/STAD/data/masks/{stem}.png")
        else:
            img = cv2.imread(old_img_path)
            mask = np.zeros(img.shape)
            cv2.imwrite(f"/dgx/github/STAD/data/masks/{stem}.png", mask)

        os.rename(old_img_path, f"/dgx/github/STAD/data/images/{stem}.png")


if __name__ == "__main__":

    df = create_info_csv()
    move_images_and_masks(df)
