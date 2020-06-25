import cv2
import albumentations as albu

from pathlib import Path
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    
    def __init__(self,
                 img_dir: Path,
                 augmentations: albu.Compose = None,
                 preprocessing: albu.Compose = None) -> None:
        
        self.img_paths = [p for p in img_dir.glob('*.png')]
        self.augmentations = augmentations
        self.preprocessing = preprocessing

        
    def __getitem__(self, 
                    idx: int):
            
            # The original image from the dataset has RGB channels, but flipped in the form of BGR. 
            # So, we use CV2 to revert back to the standard RGB.
            img_path = str(self.img_paths[idx])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.augmentations:
                sample = self.augmentations(image=img)
                img = sample['image']

            if self.preprocessing:
                sample = self.preprocessing(image=img)
                img= sample['image']

            return img
        
        
    def __len__(self) -> int:
        
        return len(self.img_paths)