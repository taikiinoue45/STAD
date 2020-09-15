from pathlib import Path

import cv2
import pandas as pd
from torch.utils.data import Dataset

import stad.typehint as T


class MVTecDataset(Dataset):
    def __init__(self, cfg: T.DictConfig, augs_dict: T.Dict[str, T.Compose], data_type: str):

        self.base = Path(cfg.dataset.base)
        self.augs = augs_dict[data_type]
        self.stem_list = []

        df = pd.read_csv(self.base / "info.csv")
        for query in cfg.dataset[data_type].query:
            stem = df.query(query)["stem"]
            self.stem_list += stem.to_list()

    def __getitem__(self, idx: int) -> dict:

        stem = self.stem_list[idx]
        img = cv2.imread(str(self.base / f"images/{stem}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.base / f"masks/{stem}.png"))

        sample = self.augs(image=img, mask=mask)
        sample["stem"] = stem
        return sample

    def __len__(self) -> int:

        return len(self.stem_list)
