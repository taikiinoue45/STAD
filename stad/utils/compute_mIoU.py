import cv2
import numpy as np

from tqdm import tqdm
from pathlib import Path


def compute_mIoU():
    
    # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
    # https://hydra.cc/docs/tutorial/working_directory    
    
    base = Path('test/anomaly')

    max_anomaly_score = -1
    for p in base.glob('*_anomaly_map.npy'):

        with open(p, 'rb') as f:
            anomaly_map = np.load(f)
            max_anomaly_score = max(max_anomaly_score, int(anomaly_map.max()))

    best_mIoU = -1
    for threshold in tqdm(np.arange(0, max_anomaly_score, 1)):

        mIoU = 0
        cnt = 0
        for p in base.glob('*_mask.png'):

            idx, _ = p.stem.split('_')

            mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            mask[mask != 0] = 1

            with open(base / f'{idx}_anomaly_map.npy', 'rb') as f:
                anomaly_map = np.load(f)
                anomaly_map[anomaly_map <  threshold] = 0
                anomaly_map[anomaly_map >= threshold] = 1
                anomaly_map = anomaly_map.astype(np.uint8)

            sum_map = anomaly_map + mask
            mIoU += np.sum(sum_map == 2) / np.sum(sum_map != 0)
            cnt += 1

        best_mIoU = max(best_mIoU, mIoU / cnt)
        
    with open('mIoU.txt', 'w') as f:
        f.write(f'max_{max_anomaly_score}\n')
        f.write(f'threshold_{threshold}\n')
        f.write(f'mIoU_{best_mIoU}\n')