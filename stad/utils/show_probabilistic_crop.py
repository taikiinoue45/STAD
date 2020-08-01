import cv2
import numpy as np
import matplotlib.pyplot as plt

from .load_log import load_log
from pathlib import Path
from mpl_toolkits.axes_grid1 import ImageGrid


def savefig(path_savefig: str, img: np.array, cumulative_heatmap: np.array, H: list, W: list):

    fig = plt.figure(figsize=(12, 4))

    # How to get two subplots to share the same y-axis with a single colorbar
    # https://stackoverflow.com/a/38940369
    grid = ImageGrid(
        fig=fig,
        rect=111,
        nrows_ncols=(1, 3),
        axes_pad=0.15,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.15,
    )

    grid[0].imshow(img)
    grid[0].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    grid[1].imshow(img)
    grid[1].imshow(cumulative_heatmap, alpha=0.3, cmap="jet")
    grid[1].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)

    grid[2].imshow(img)
    im = grid[2].imshow(cumulative_heatmap, alpha=0.3, cmap="jet")
    grid[2].scatter(W, H, s=1, color="k")
    grid[2].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    grid[2].cax.colorbar(im)
    grid[2].cax.toggle_label(True)

    plt.savefig(path_savefig, bbox_inches="tight")
    plt.close()


def show_probabilistic_crop(cfg):

    # CWD is STAD/stad/outputs/yyyy-mm-dd/hh-mm-ss
    # https://hydra.cc/docs/tutorial/working_directory
    base = Path(cfg.dataset.val)
    log = load_log()
    cumulative_heatmap = np.array([])
    for p in Path(".").glob("* - * - val - *.npy"):
        epoch, idx, _, stem = p.stem.split(" - ")

        for img_path in base.glob(f"images/{stem}*"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        with open(p, "rb") as f:

            if len(cumulative_heatmap) == 0:
                cumulative_heatmap = np.load(f)
            else:
                cumulative_heatmap += np.load(f)

        path_savefig = f"{epoch} - {idx} - val - {stem} - prob_crop.png"
        epoch = str(int(epoch) + 1)
        H = log[epoch]["h"]
        W = log[epoch]["w"]
        savefig(path_savefig, img, cumulative_heatmap, H, W)
