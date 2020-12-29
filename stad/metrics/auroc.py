import matplotlib.pyplot as plt
from numpy import ndarray as NDArray
from sklearn.metrics import roc_auc_score, roc_curve


def compute_auroc(epoch: int, all_heatmaps: NDArray, all_masks: NDArray) -> float:

    num_data, h, w = all_heatmaps.shape
    y_score = all_heatmaps.reshape(num_data, -1).max(axis=1)  # y_score.shape -> (num_data,)
    y_true = all_masks.reshape(num_data, -1).max(axis=1)  # y_true.shape -> (num_data,)

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    plt.plot(fpr, tpr, marker="o", color="k")
    plt.savefig(f"epochs/{epoch}/roc_curve.png")

    return roc_auc_score(y_true, y_score)
