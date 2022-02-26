"""Here are some logical functions.

Expected, input tensors must contain only 0 or 1.
"""


import torch


def logical_or(tensor_a, tensor_b):
    """Calculate logical OR between tensor_a and tensor_b.

    Args:
        tensor_a ():
        tensor_b ():

    Returns:
        Logical OR between tensor_a and tensor_b, same shape as input tensors
    """
    return torch.bitwise_or(tensor_a, tensor_b)


def logical_and(tensor_a, tensor_b):
    """Calculate logical AND between tensor_a and tensor_b.

    Args:
        tensor_a ():
        tensor_b ():

    Returns:
        Logical AND between tensor_a and tensor_b, same shape as input tensors
    """
    return tensor_a * tensor_b


def logical_not(tensor):
    """Calculate logical NOT from input tensor.

    Args:
        tensor ():

    Returns:
        Logical NOT from input tensor, same shape as input tensor
    """
    return tensor.neg() + 1
