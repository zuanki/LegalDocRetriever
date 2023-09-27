import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(batch_pred: torch.Tensor, batch_target: torch.Tensor) -> float:
    """
    Computes the accuracy for a batch of predictions and targets

    Args:
        batch_pred (torch.Tensor): Batch of predictions
        batch_target (torch.Tensor): Batch of targets

    Returns:
        float: Accuracy
    """
    batch_pred = torch.argmax(batch_pred, dim=1)
    batch_target = batch_target

    return (batch_pred == batch_target).sum().item() / len(batch_pred)


def f1_score(batch_pred: torch.Tensor, batch_target: torch.Tensor) -> float:
    """
    Computes the f1 score for a batch of predictions and targets

    Args:
        batch_pred (torch.Tensor): Batch of predictions
        batch_target (torch.Tensor): Batch of targets

    Returns:
        float: F1 score
    """
    batch_pred = torch.argmax(batch_pred, dim=1)
    batch_target = batch_target

    return f1_score(batch_pred, batch_target)
