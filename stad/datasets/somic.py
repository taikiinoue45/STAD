from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from numpy import ndarray as NDArray
from torch import Tensor
from torch.utils.data import Dataset
from typing_extensions import Literal

from stad.transforms import Compose


class SomicDataset(Dataset):
    def __init__(
        self,
        root: Union[Path, str],
        color_type: Literal["color", "gray"],
        labelname_to_label: Dict[str, int],
        query_list: List[str],
        preprocess: Compose,
        debug: bool,
    ) -> None:

        """
        Args:
            root (Union[Path, str]): Path to */somic-data/dataset directory
            labelname_to_label (Dict[str, int]): Dict to convert label name to label
            query_list (List[str]): Query list to extract arbitrary rows from info.csv
            preprocess (Composite): List of transforms
            debug (bool): If true, preprocessed images are saved
        """

        self.root = Path(root)
        self.color_type = color_type
        self.labelname_to_label = labelname_to_label
        self.preprocess = preprocess
        self.debug = debug

        df = pd.read_csv(self.root / "info.csv")
        self.stem_list = []
        for q in query_list:
            self.stem_list += df.query(q)["stem"].to_list()

        self.default_labelname_to_label = {
            "kizu_dakon": 1,
            "kizu_ware": 2,
            "kizu_zairyou": 3,
            "ignore_shallow": 4,
            "ignore_cutting": 5,
            "ignore_oil": 6,
        }

    def _load_img(self, index: int) -> NDArray:

        img_path = self.root / f"{self.color_type}_images/{self.stem_list[index]}.jpg"
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        assert len(img.shape) == 3
        return img

    def _load_mask(self, index: int) -> NDArray:

        mask_path = self.root / f"masks/{self.stem_list[index]}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert len(mask.shape) == 2
        return mask

    def _update_mask_label(self, mask: NDArray) -> NDArray:

        for labelname, label in self.labelname_to_label.items():
            default_label = self.default_labelname_to_label[labelname]
            if label != default_label:
                mask[mask == default_label] = label
        return mask

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

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:

        img = self._load_img(index)
        mask = self._load_mask(index)
        mask = self._update_mask_label(mask)

        data_dict = self.preprocess(image=img, mask=mask)

        if self.debug:
            self._save_transformed_images(index, data_dict["image"], data_dict["mask"])

        return (data_dict["Image"], data_dict["Mask"])

    def __len__(self) -> int:

        return len(self.stem_list)
