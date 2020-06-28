import cv2
import albumentations as albu

from pathlib import Path
from torch.utils.data import Dataset


class MVTecDataset(Dataset):

    def __init__(self,
                 img_dir: Path,
                 augs: albu.Compose = None) -> None:

        self.img_paths = [p for p in img_dir.glob('*.png')]
        self.augs = augs

    def __getitem__(self,
                    idx: int):

        img_path = str(self.img_paths[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augs:
            sample = self.augs(image=img)
            img = sample['image']

        return img

    def __len__(self) -> int:

        return len(self.img_paths)
