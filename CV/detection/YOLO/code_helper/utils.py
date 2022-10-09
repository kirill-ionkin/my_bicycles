import numpy as np

from code_helper import metrics


def convert_x1y1wh_to_x1y1x2y2(coords: np.ndarray) -> np.ndarray:
    """Conver (x1, y1, w, h, confidence_score, class) format to (x1, y1, x1,
    y2, confidence_score, class)

    Args:
        coords: Coordinates info in format (x1, y1, w, h, confidence_score, class). corrds.shape = (n, 6), where n - Number of examples

    Returns:
        Coordinates info in format (x1, y1, x1, y2, confidence_score, class).
    """
    x1 = coords[:, 0:1]
    y1 = coords[:, 1:2]
    width = coords[:, 2:3]
    height = coords[:, 3:4]
    confidence = coords[:, 4:5]
    class_ = coords[:, 5:6]

    x2 = x1 + width
    y2 = y1 + height

    return np.concatenate((x1, y1, x2, y2, confidence, class_), axis=1)


def convert_x1y1x2y2_to_x1y1wh(coords):
    """Conver (x1, y1, x1, y2, confidence_score, class) format to (x1, y1, w,
    h, confidence_score, class)

    Args:
        coords: Coordinates info in format (x1, y1, x1, y2, confidence_score, class). corrds.shape = (n, 6), where n - Number of examples

    Returns:
        Coordinates info in format (x1, y1, w, h, confidence_score, class).
    """
    x1 = coords[:, 0:1]
    y1 = coords[:, 1:2]
    x2 = coords[:, 2:3]
    y2 = coords[:, 3:4]
    confidence = coords[:, 4:5]
    class_ = coords[:, 5:6]

    width = x2 - x1
    height = y2 - y1

    return np.concatenate((x1, y1, width, height, confidence, class_), axis=1)


def nms(
    bboxes: np.ndarray,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    confidence_col: int = 4,
    class_col: int = 5,
) -> np.ndarray:
    """Non Maximum Suppression implementation algorithm.

    Args:
        bboxes: All predicted bboxes, bboxes.shape = (number_predicted_bboxes, 6), where each bbox containes info like (x1, y1, x2, y2, confidense, class)
        iou_threshold: Intersection over Union threshold. Defaults to 0.5.
        confidence_threshold: Confidence threshold to drop non-confident predicted bboxes. Defaults to 0.3.
        confidence_col: Number of column, which containes confidence score info. Defaults to 4.
        class_col: Number of column, which containes info, what's class was predicted inside each bbox. Defaults to 5.

    Returns:
        Predicted bboxes after Non Maximum Suppression algorithm
    """

    if not len(bboxes):
        return np.array([])

    # Drop bbox, which has less confidence than confidence_threshold
    bboxes = bboxes[bboxes[:, confidence_col] > confidence_threshold]
    # Sort bboxes by confidence score on descending order
    bboxes = bboxes[np.argsort(bboxes[:, confidence_col])[::-1]]

    indexes = np.arange(len(bboxes))
    for i, box in enumerate(bboxes):
        if i not in indexes:
            continue

        # Create required shape: box.shape = (1, 6)
        box = box[None, :]

        other_bboxes_indexes = indexes[indexes != i]
        other_bboxes = bboxes[other_bboxes_indexes]

        # Get indexes, where other boxes class has same class, as box class
        other_bboxes_same_class_indexes = other_bboxes_indexes[
            other_bboxes[:, class_col] == box[:, class_col]
        ]
        other_bboxes_same_class = other_bboxes[other_bboxes_same_class_indexes[:]]

        if len(other_bboxes_same_class):
            iou = metrics.IoU(ground_truth=box, pred_bbox=other_bboxes_same_class)

            if len(iou):
                indexes_to_drop = other_bboxes_same_class_indexes[
                    iou[:, 0] > iou_threshold
                ]

                indexes = np.setdiff1d(indexes, indexes_to_drop)
    return bboxes[indexes]
