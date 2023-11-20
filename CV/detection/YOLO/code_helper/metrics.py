import numpy as np

import torch
import torch.nn as nn


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Metric Precision at K.

    Args:
        y_true: 1-dim np.ndarray, true labels
        y_pred: 1-dim np.ndarray, predicted labels
        k:

    Returns:
        p@K metric
    """
    desc_order_y_pred = np.argsort(-y_pred)
    y_true_desc_order = y_true[desc_order_y_pred]

    return y_true_desc_order[:k].sum() / k


def average_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    precisions_at_k = []
    for k_ in range(1, k + 1):
        p_at_k = precision_at_k(y_true=y_true, y_pred=y_pred, k=k_)
        precisions_at_k.append(p_at_k)

    return sum(precisions_at_k) / k


def mean_average_precision_at_k():
    pass


def get_coordinates(bbox_coordinates: np.ndarray):
    """Return x1, y1, x2, y2 coordinates, where bbox_coordinates containes info
    x1, y1, x2, y2.

    Args:
        bbox_coordinates: bbox_coordinates.shape = (batch_size, 4)
    """
    x1 = bbox_coordinates[:, 0:1]
    y1 = bbox_coordinates[:, 1:2]
    x2 = bbox_coordinates[:, 2:3]
    y2 = bbox_coordinates[:, 3:4]

    return x1, y1, x2, y2


def intersection(ground_truth: np.ndarray, pred_bbox: np.ndarray) -> np.ndarray:
    """Calculate intersection area between ground_truth and pred_bbox, where
    both contain info x1, y1, x2, y2.

    Args:
        ground_truth: ground_truth.shape = (batch_size, 4)
        pred_bbox: predicted bounding boxes, pred_bbox.shape = (batch_size, 4)
    """

    x1_pred, y1_pred, x2_pred, y2_pred = get_coordinates(pred_bbox)
    x1_true, y1_true, x2_true, y2_true = get_coordinates(ground_truth)

    x1 = np.maximum(x1_pred, x1_true)
    y1 = np.maximum(y1_pred, y1_true)
    x2 = np.minimum(x2_pred, x2_true)
    y2 = np.minimum(y2_pred, y2_true)

    width = x2 - x1
    hight = y2 - y1
    return np.where(width > 0, width, 0) * np.where(hight > 0, hight, 0)


def union(ground_truth: np.ndarray, pred_bbox: np.ndarray) -> np.ndarray:
    """Calculate union area between ground_truth and pred_bbox, where both
    contain info x1, y1, x2, y2.

    Args:
        ground_truth: ground_truth.shape = (batch_sixe, 4)
        pred_bbox: predicted bounding boxes, pred_bbox.shape = (batch_size, 4)
    """
    x1_pred, y1_pred, x2_pred, y2_pred = get_coordinates(pred_bbox)
    x1_true, y1_true, x2_true, y2_true = get_coordinates(ground_truth)

    pred_bbox_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    ground_truth_area = (x2_true - x1_true) * (y2_true - y1_true)

    return (
        pred_bbox_area
        + ground_truth_area
        - intersection(ground_truth=ground_truth, pred_bbox=pred_bbox)
    )


def IoU(ground_truth: np.ndarray, pred_bbox: np.ndarray, epsilon=1e-15) -> np.ndarray:
    """Calculate IoU(Intersection over Union) metric between ground_truth and
    pred_bbox, where both contain info x1, y1, x2, y2.

    Args:
        ground_truth: ground_truth.shape = (batch_sixe, 4)
        pred_bbox: predicted bounding boxes, pred_bbox.shape = (batch_size, 4)

    Returns:
        _description_
    """
    union_area = union(ground_truth=ground_truth, pred_bbox=pred_bbox)
    intersection_area = intersection(ground_truth=ground_truth, pred_bbox=pred_bbox)

    return intersection_area / (union_area + epsilon)
