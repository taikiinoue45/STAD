import cv2
import numpy as np
import matplotlib.pyplot as plt

from .load_log import load_log
from pathlib import Path


def savefig(path_savefig: Path,
            img: np.array,
            cumulative_anomaly_map: np.array, 
            H: list,
            W: list):
    
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
    plt.imshow(cumulative_anomaly_map, alpha=0.5)
    plt.title('Cumulative Anomaly Map', fontsize=20)
    plt.xticks(color='None')
    plt.yticks(color='None') 
    plt.tick_params(length=0)
    plt.tight_layout()

    plt.subplot(133)
    plt.imshow(img)
    plt.imshow(cumulative_anomaly_map, alpha=0.5)
    plt.scatter(W, H, s=0.5, color='r')
    plt.title('Probabilistic Crop', fontsize=20)
    plt.xticks(color='None')
    plt.yticks(color='None') 
    plt.tick_params(length=0)
    plt.tight_layout()    

    plt.savefig(path_savefig)
    plt.close()

    
    
def show_probabilistic_crop():
    
    
    log = load_log()
    cumulative_anomaly_map = np.array([])

    base = Path('.')
    for p in base.glob('val/*_img.jpg'):
        
        epoch, idx, _ = p.stem.split('_')
        
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(base / f'val/{epoch}_{idx}_anomaly_map.npy', 'rb') as f:
            
            if len(cumulative_anomaly_map) == 0:
                cumulative_anomaly_map = np.load(f)
            else:
                cumulative_anomaly_map += np.load(f)

        path_savefig = base / f'val/{epoch}_probabilistic_crop.png'
        epoch = str(int(epoch)+1)
        H = log[epoch]['h']
        W = log[epoch]['w']
        savefig(path_savefig, img, cumulative_anomaly_map, H, W)
