from pathlib import Path
from typing import List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset

from stad.transforms import Compose


class MVTecDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[Path, str],
        query_list: List[str],
        preprocess: Compose,
        debug: bool,
    ) -> None:

        """
        Args:
            data_dir (Union[Path, str]): Path to directory with info.csv, images/ and masks/
            query_list (List[str]): Query list to extract arbitrary rows from info.csv
            preprocess (Composite): List of transforms
            debug (bool): If true, preprocessed images are saved
        """

        self.data_dir = Path(data_dir)
        self.preprocess = preprocess
        self.debug = debug

        df = pd.read_csv(self.data_dir / "info.csv")
        self.stem_list = []
        for q in query_list:
            self.stem_list += df.query(q)["stem"].to_list()

    def __getitem__(self, index: int) -> Tuple[str, Tensor, Tensor]:

        stem = self.stem_list[index]

        img_path = str(self.data_dir / f"images/{stem}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask_path = str(self.data_dir / f"masks/{stem}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask != 0] = 1

        data_dict = self.preprocess(image=img, mask=mask)

        if self.debug:
            self._save_transformed_images(index, data_dict["image"], data_dict["mask"])

        return (img_path, data_dict["image"], data_dict["mask"])

    def _save_transformed_images(self, index: int, img: Tensor, mask: Tensor) -> None:

        img = img.permute(1, 2, 0).detach().numpy()
        mask = mask.detach().numpy()
        plt.figure(figsize=(9, 3))

        plt.subplot(131)
        plt.title("Input Image")
        plt.imshow(img)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(132)
        plt.title("Ground Truth Mask")
        plt.imshow(mask)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.subplot(133)
        plt.title("Supervision")
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        plt.tight_layout()
        plt.savefig(f"{self.stem_list[index]}.png")

    def __len__(self) -> int:

        return len(self.stem_list)
