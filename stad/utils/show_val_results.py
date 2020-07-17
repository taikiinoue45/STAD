import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm


def savefig(path_savefig: Path,
            img: np.array, 
            mask: np.array, 
            anomaly_map: np.array):
    
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.imshow(img)
    plt.title('Original Image', fontsize=20)
    plt.xticks(color='None')
    plt.yticks(color='None') 
    plt.tick_params(length=0)
    plt.tight_layout()

    plt.subplot(132)
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5)
    plt.title('Ground Truth', fontsize=20)
    plt.xticks(color='None')
    plt.yticks(color='None') 
    plt.tick_params(length=0)
    plt.tight_layout()

    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(anomaly_map, alpha=0.5)
    plt.title('Anomaly Map', fontsize=20)
    plt.xticks(color='None')
    plt.yticks(color='None') 
    plt.tick_params(length=0)
    plt.tight_layout()    

    plt.savefig(path_savefig)
    plt.close()
    

    
def show_val_results(base: Path):
    
    # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
    # https://hydra.cc/docs/tutorial/working_directory
    
    # Get the maximum anomaly score in all anomaly maps
    # It is used to make the maximum anomaly score same over all anomaly maps
    max_anomaly_score = -1
    for p in Path('test/anomaly').glob('*_anomaly_map.npy'):

        with open(p, 'rb') as f:
            anomaly_map = np.load(f)
            max_anomaly_score = max(max_anomaly_score, anomaly_map.max())

            
    # Show and save the anomaly maps 
    for p in tqdm(base.glob('*_img.jpg')):

        epoch, idx, _ = p.stem.split('_')

        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(base / f'{epoch}_{idx}_mask.png'))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        with open(base / f'{epoch}_{idx}_anomaly_map.npy', 'rb') as f:
            anomaly_map = np.load(f)
            anomaly_map[0, 0] = max_anomaly_score

        path_savefig = base / f'{epoch}_{idx}_results.png'
        savefig(path_savefig, img, mask, anomaly_map)



