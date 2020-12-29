import logging

import numpy as np
from albumentations.core.transforms_interface import DualTransform


log = logging.getLogger(__name__)


class ProbabilisticCrop(DualTransform):
    def __init__(
        self,
        height: int,
        width: int,
        heatmap: np.array = np.array([]),
        always_apply: bool = False,
        p: float = 1.0,
    ):

        super(ProbabilisticCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.half_h = height // 2
        self.half_w = width // 2
        self.heatmap = heatmap

    def apply(self, img, **params):

        if len(self.heatmap) == 0:
            self.heatmap = np.ones(img.shape[:2])

        a = np.arange(np.prod(self.heatmap.shape))
        p = self.heatmap.flatten()
        p = p / p.sum()
        sample = np.random.choice(a=a, size=1, p=p)[0]

        h, w = divmod(sample, self.heatmap.shape[1])

        logging.info(f"h - {h}")
        logging.info(f"w - {w}")

        h = max(h, self.half_h)
        h = min(h, img.shape[0] - self.half_h)
        w = max(w, self.half_w)
        w = min(w, img.shape[1] - self.half_w)

        return img[h - self.half_h : h + self.half_h, w - self.half_w : w + self.half_w, :]

    def get_transform_init_args_names(self):
        return ("height", "width")
