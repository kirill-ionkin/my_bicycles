if __name__ != "__main__":
    import numpy as np
    import scipy as sp
    import pandas as pd

    import torch
    import sklearn


def calculate_accuracy(y_preds,
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
    return sum(torch.argmax(y_preds, dim=1) == y_true).item() / len(y_true)


def calculate_f1_score(y_preds,
                       y_true,
                       logits=True,
                       threshold=None
                      ):
    '''
    '''


def calculate_precision(y_preds,
                        y_true, 
                        logits=True
                        ):
    """

    """
    pass


def calculate_recall(y_preds,
                     y_true, 
                     logits=True
                     ):
    """

    """
    pass


def calculate_equal_error_rate(y_preds,
                               y_true,
                               logits=True
                              ):
    """
    
    """
    if logits:
        if len(y_preds.size()) == 2:
            y_preds = torch.nn.functional.softmax(y_preds, dim=1)[:, 1]
        else:
            y_preds = torch.nn.functional.sigmoid(y_preds)
    else:
        if len(y_preds.size()) == 2:
            y_preds = y_preds[:, 1]
        else:
            pass
    y_true = y_true.numpy()
    y_preds = y_preds.numpy()
    
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_preds)

    eer = sp.optimize.brentq(lambda x : 1.0 - x - sp.interpolate.interp1d(fpr, tpr)(x), 0.0 , 1.0)
    #thresh = sp.interpolate.interp1d(fpr, thresholds)(eer)
    return eer


