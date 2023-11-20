import torch


def get_coordinates(bbox_coordinates: torch.Tensor):
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


def intersection(ground_truth: torch.Tensor, pred_bbox: torch.Tensor) -> torch.Tensor:
    """Calculate intersection area between ground_truth and pred_bbox, where
    both contain info x1, y1, x2, y2.

    Args:
        ground_truth: ground_truth.shape = (batch_sixe, 4)
        pred_bbox: predicted bounding boxes, pred_bbox.shape = (batch_size, 4)
    """
    x1_pred, y1_pred, x2_pred, y2_pred = get_coordinates(pred_bbox)
    x1_true, y1_true, x2_true, y2_true = get_coordinates(ground_truth)

    x1 = torch.max(x1_pred, x1_true)
    y1 = torch.max(y1_pred, y1_true)
    x2 = torch.min(x2_pred, x2_true)
    y2 = torch.min(y2_pred, y2_true)

    width = x2 - x1
    hight = y2 - y1
    return torch.clamp(width, min=0) * torch.clamp(hight, min=0)


def union(ground_truth: torch.Tensor, pred_bbox: torch.Tensor) -> torch.Tensor:
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


def IoU(
    ground_truth: torch.Tensor, pred_bbox: torch.Tensor, epsilon=1e-15
) -> torch.Tensor:
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
