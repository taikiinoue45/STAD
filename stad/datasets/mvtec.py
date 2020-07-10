import cv2
import numpy as np
import albumentations as albu

from pathlib import Path
from torch.utils.data import Dataset


class MVTecDataset(Dataset):

    def __init__(self,
                 img_dir: Path,
                 mask_dir: Path,
                 augs: albu.Compose,
                 is_anomaly: bool) -> None:

        self.img_paths = []
        self.mask_paths = []

        for img_path in img_dir.glob('*.png'):
            mask_path = mask_dir / f'{img_path.stem}_mask.png'
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)

        self.augs = augs
        self.is_anomaly = is_anomaly


    def __getitem__(self,
                    idx: int):

        img_path = str(self.img_paths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        arr = img

        if self.is_anomaly:
            mask_path = str(self.mask_paths[idx])
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        else:
            mask = np.zeros(img.shape)

        if self.augs:
            sample = self.augs(image=img)
            tsr = sample['image']
            
        return tsr, arr, mask

    def __len__(self) -> int:

        return len(self.img_paths)
