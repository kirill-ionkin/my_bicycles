import numpy as np
import scipy as sp
import pandas as pd

import torch
import sklearn

from logical_functions import logical_and, logical_or, logical_not


def calculate_TP(y_preds, y_true):
    """
    Calculate True Positive value from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels

    Returns:
        True Positive value
    """

    return logical_and(y_true, y_preds).sum().item()


def calculate_TN(y_preds, y_true):
    """
    Calculate True Negative value from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels

    Returns:
        True Negative value
    """

    return logical_and(logical_not(y_true), logical_not(y_preds)).sum().item()


def calculate_FP(y_preds, y_true):
    """
    Calculate False Positive value from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels

    Returns:
        False Positive value
    """

    return y_preds[y_true == 0].sum().item()


def calculate_FN(y_preds, y_true):
    """
    Calculate False Negative value from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels

    Returns:
        False Negative value
    """

    return logical_not(y_preds[y_true == 1]).sum().item()


def calculate_TP_TN_FP_FN(y_preds, y_true):
    """
    Calculate all standard metrics: True Positive, True Negative, False Positive, False Negative
    Args:
        y_preds (): predictions labels
        y_true (): true labels

    Returns:
        True Positive, True Negative, False Positive, False Negative values
    """

    TP = calculate_TP(y_preds, y_true)
    TN = calculate_TN(y_preds, y_true)
    FP = calculate_FP(y_preds, y_true)
    FN = calculate_FN(y_preds, y_true)
    return TP, TN, FP, FN


def calculate_accuracy(y_preds,
                       y_true,
                       logits=True
                      ):
    """
    Calculate accuracy from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels
        logits (): Depends on model architecture. True, if model return logits or False, if model return probability

    Returns:
        Accuracy value
    """

    if len(y_preds.size()) == 2:
        y_preds = torch.argmax(y_preds, dim=1)

    TP, TN, FP, FN = calculate_TP_TN_FP_FN(y_preds, y_true)
    return (TP + TN) / (TP + TN + FP + FN)


def calculate_precision(y_preds,
                        y_true, 
                        logits=True
                        ):
    """
    Calculate precision from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels
        logits (): Depends on model architecture. True, if model return logits or False, if model return probability

    Returns:
        Precision value
    """

    if len(y_preds.size()) == 2:
        y_preds = torch.argmax(y_preds, dim=1)

    TP, TN, FP, FN = calculate_TP_TN_FP_FN(y_preds, y_true)
    return TP / (TP + FP) if (TP + FP) else 0.0


def calculate_recall(y_preds,
                     y_true, 
                     logits=True
                     ):
    """
    Calculate recall from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels
        logits (): Depends on model architecture. True, if model return logits or False, if model return probability

    Returns:
        Recall value
    """

    if len(y_preds.size()) == 2:
        y_preds = torch.argmax(y_preds, dim=1)

    TP, TN, FP, FN = calculate_TP_TN_FP_FN(y_preds, y_true)
    return TP / (TP + FN) if (TP + FN) else 0.0


def calculate_f1_score(y_preds,
                       y_true,
                       logits=True,
                       threshold=None
                       ):
    """
    Calculate f1_score from predictions and true labels

    Args:
        y_preds (): predictions labels
        y_true (): true labels
        logits (): Depends on model architecture. True, if model return logits or False, if model return probability
        threshold ():

    Returns:
        F1-score value
    """

    precision = calculate_precision(y_preds, y_true)
    recall = calculate_recall(y_preds, y_true)

    if precision and recall:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def calculate_equal_error_rate(y_preds,
                               y_true,
                               logits=True
                              ):
    """
    Args:
        y_preds ():
        y_true ():
        logits ():

    Returns:

    """

    if logits:
        if len(y_preds.size()) == 2:
            y_preds = torch.nn.functional.softmax(y_preds, dim=1)[:, 1]  # probability of positive class
        else:
            y_preds = torch.nn.functional.sigmoid(y_preds)
    else:
        if len(y_preds.size()) == 2:
            y_preds = y_preds[:, 1]  # probability of positive class
        else:
            pass

    y_true = y_true.numpy()
    y_preds = y_preds.numpy()
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_preds)

    eer = sp.optimize.brentq(lambda x : 1.0 - x - sp.interpolate.interp1d(fpr, tpr)(x), 0.0, 1.0)
    #thresh = sp.interpolate.interp1d(fpr, thresholds)(eer)
    return eer


if __name__ == "__main__":
    pass
