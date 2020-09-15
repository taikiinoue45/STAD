from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def compute_mIoU(cfg) -> None:

    # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
    # https://hydra.cc/docs/tutorial/working_directory

    base = Path(cfg.dataset.base)
    max_anomaly_score = -1
    for p in Path(".").glob("* - test_anomaly - *.npy"):

        with open(p, "rb") as f:
            heatmap = np.load(f)
            max_anomaly_score = max(max_anomaly_score, heatmap.max())

    best_mIoU = -1.0
    best_threshold = -1.0
    eps = 10 ** -5
    for threshold in tqdm(np.linspace(0, max_anomaly_score, 1000)):

        IoU_li = []
        for p in Path(".").glob("* - test_anomaly - *.npy"):
            idx, _, stem = p.stem.split(" - ")

            for mask_path in base.glob(f"masks/{stem}*"):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                mask[mask != 0] = 1

            with open(p, "rb") as f:
                heatmap = np.load(f)
                heatmap[heatmap < threshold] = 0
                heatmap[heatmap >= threshold] = 1
                heatmap = heatmap.astype(np.uint8)

            intersection = np.sum(heatmap + mask == 2)
            union = np.sum(heatmap + mask != 0)
            IoU_li.append(intersection / (union + eps))

        mIoU = sum(IoU_li) / (len(IoU_li) + eps)

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_threshold = threshold

    with open("mIoU.txt", "w") as f:
        f.write(f"max anomaly score - {max_anomaly_score}\n")
        f.write(f"threshold - {best_threshold}\n")
        f.write(f"mIoU - {best_mIoU}\n")
