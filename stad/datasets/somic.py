import cv2
import albumentations as albu

from pathlib import Path
from torch.utils.data import Dataset


class SomicDataset(Dataset):
    def __init__(self, base: Path, augs: albu.Compose) -> None:

        self.img_paths = [str(p) for p in base.glob("images/*.bmp")]
        self.augs = augs

    def __getitem__(self, idx: int):

        img = cv2.imread(self.img_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augs:
            sample = self.augs(image=img)

        sample["img_path"] = self.img_paths[idx]
        return sample

    def __len__(self) -> int:

        return len(self.img_paths)
