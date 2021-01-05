from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from numpy import ndarray as NDArray


def savefig(
    epoch: int,
    all_img_paths: List[str],
    all_heatmaps: List[NDArray],
    all_masks: List[NDArray],
) -> None:

    for i, (img_path, heatmap, mask) in enumerate(zip(all_img_paths, all_heatmaps, all_masks)):

        # How to get two subplots to share the same y-axis with a single colorbar
        # https://stackoverflow.com/a/38940369
        grid = ImageGrid(
            fig=plt.figure(figsize=(12, 4)),
            rect=111,
            nrows_ncols=(1, 3),
            axes_pad=0.15,
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.15,
        )

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        grid[0].imshow(img)
        grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        grid[1].imshow(img)
        grid[1].imshow(mask, alpha=0.3, cmap="Reds")
        grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

        grid[2].imshow(img)
        im = grid[2].imshow(heatmap, alpha=0.3, cmap="jet")
        grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
        grid[2].cax.colorbar(im)
        grid[2].cax.toggle_label(True)

        plt.savefig(f"epochs/{epoch}/{Path(img_path).stem}.png", bbox_inches="tight")
        plt.close()
