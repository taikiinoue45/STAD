import cv2
import numpy as np
import albumentations as albu

from pathlib import Path
from torch.utils.data import Dataset


class SomicDataset(Dataset):

    def __init__(self,
                 data_dir: Path,
                 augs: albu.Compose) -> None:
        
        self.img_paths = []
        self.mask_paths = []

        for img_path in data_dir.glob('images/*.bmp'):
            mask_path = data_dir / f'masks/{img_path.stem}.png'
            self.img_paths.append(img_path)
            self.mask_paths.append(mask_path)

        self.augs = augs
        

    def __getitem__(self,
                    idx: int):

        img_path = str(self.img_paths[idx])
        raw_img = cv2.imread(img_path)
        raw_img = raw_img[100:1050, :, :]
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        
        mask_path = str(self.mask_paths[idx])
        mask = cv2.imread(mask_path)
        mask = mask[100:1050, :, :]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        if self.augs:
            sample = self.augs(image=img, mask=mask)
            img = sample['image']
            mask = sample['mask']
            
        return img, raw_img, mask
    

    def __len__(self) -> int:

        return len(self.img_paths)
