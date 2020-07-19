import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path


def compute_mIoU():
    
    # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
    # https://hydra.cc/docs/tutorial/working_directory    
    
    base = Path('.')
    max_anomaly_score = -1
    for p in base.glob('* - test_anomaly_heatmap.npy'):

        with open(p, 'rb') as f:
            heatmap = np.load(f)
            max_anomaly_score = max(max_anomaly_score, heatmap.max())

    best_mIoU = -1
    best_threshold = -1
    for threshold in tqdm(np.linspace(0, max_anomaly_score, 1000)):

        sum_IoU = 0
        cnt = 0
        for p in base.glob('* - test_anomaly_mask.png'):

            idx, _ = p.stem.split(' - ')

            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            mask[mask != 0] = 1

            with open(base / f'{idx} - test_anomaly_heatmap.npy', 'rb') as f:
                heatmap = np.load(f)
                heatmap[heatmap <  threshold] = 0
                heatmap[heatmap >= threshold] = 1
                heatmap = heatmap.astype(np.uint8)

            math_set = heatmap + mask
            sum_IoU += np.sum(math_set == 2) / np.sum(math_set != 0)
            cnt += 1

        mIoU = sum_IoU / cnt

        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_threshold = threshold
        
    with open('mIoU.txt', 'w') as f:
        f.write(f'max_{max_anomaly_score}\n')
        f.write(f'threshold_{best_threshold}\n')
        f.write(f'mIoU_{best_mIoU}\n')