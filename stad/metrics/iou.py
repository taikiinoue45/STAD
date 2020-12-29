from typing import List

import numpy as np
from numpy import ndarray as NDArray


def compute_iou(heatmap_list: List[NDArray], mask_list: List[NDArray]) -> float:

    for threshold in range(10):

        segmap_list
        total_area_dict = _intersect_and_union(heatmap, mask)
        iou_per_class = total_area_dict["intersect"] / total_area_dict["union"]

        iou_dict = {}
        iou_dict["mean"] = iou_per_class.mean()
        for class_id, class_iou in enumerate(iou_per_class):
            iou_dict[str(class_id)] = class_iou

    return iou_dict


def _intersect_and_union(segmap, mask):

    zero_array = np.zeros(num_classes, dtype=np.float)
    total_area_dict = {
        "segmap": zero_array.copy(),
        "mask": zero_array.copy(),
        "intersect": zero_array.copy(),
        "union": zero_array.copy(),
    }

    for segmap, mask in zip(segmap, mask):

        segmap = segmap[bool_array]
        intersect = segmap[segmap == mask]

        bins = np.arange(num_classes + 1)
        segmap_area, _ = np.histogram(segmap, bins=bins)
        mask_area, _ = np.histogram(mask, bins=bins)
        intersect_area, _ = np.histogram(intersect, bins=bins)
        union_area = segmap_area + mask_area - intersect_area

        total_area_dict["segmap"] += segmap_area
        total_area_dict["mask"] += mask_area
        total_area_dict["intersect"] += intersect_area
        total_area_dict["union"] += union_area

    return total_area_dict
