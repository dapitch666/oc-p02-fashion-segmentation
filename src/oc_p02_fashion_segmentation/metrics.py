"""
Metrics to compare a predicted segmentation mask with a reference mask.

The IoU is split in two steps, so the pixel counts of several images can be
summed before computing a single IoU over a whole dataset:
intersection_and_union counts the pixels, iou_from_counts turns the counts
into IoU values.
"""

import numpy as np

from oc_p02_fashion_segmentation.constants import NUM_CLASSES


def intersection_and_union(
    true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = NUM_CLASSES
) -> tuple[np.ndarray, np.ndarray]:
    """
    Count the pixels in the intersection and in the union of each class.

    Args:
        true_mask (np.ndarray): 2D reference mask with class indices.
        pred_mask (np.ndarray): 2D predicted mask with class indices,
            same shape as true_mask.
        num_classes (int): Number of classes, with IDs from 0 to
            num_classes - 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: Two int arrays of num_classes values.
            The value at index c is, for class c, the number of pixels labelled
            c in both masks (intersection), and in at least one of them (union).

    Raises:
        ValueError: If the masks do not have the same shape.
    """
    if true_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Masks must have the same shape, got {true_mask.shape} "
            f"and {pred_mask.shape}"
        )

    intersection = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        true_class = true_mask == class_id
        pred_class = pred_mask == class_id
        intersection[class_id] = np.count_nonzero(true_class & pred_class)
        union[class_id] = np.count_nonzero(true_class | pred_class)
    return intersection, union


def iou_from_counts(intersection: np.ndarray, union: np.ndarray) -> np.ndarray:
    """
    Convert intersection and union pixel counts into IoU values.

    Args:
        intersection (np.ndarray): Pixels in the intersection of each class.
        union (np.ndarray): Pixels in the union of each class, same shape.

    Returns:
        np.ndarray: Array of floats, the IoU of each class. It is nan where the
            union is 0, i.e. for a class missing from both masks, since the IoU
            is then undefined.
    """
    iou = np.full(union.shape, np.nan)
    np.divide(intersection, union, out=iou, where=union > 0)
    return iou


def iou_by_class(
    true_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """
    Compute the IoU of each class between a reference and a predicted mask.

    Shortcut for intersection_and_union followed by iou_from_counts.
    A class missing from both masks gets nan.
    """
    return iou_from_counts(*intersection_and_union(true_mask, pred_mask, num_classes))
