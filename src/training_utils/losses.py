import torch
import torch.nn as nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        # self.weight = torch.tensor([0.75, 0.25])

    def forward(self, logits, target):
        loss = nn.CrossEntropyLoss(weight=self.weight)
        return loss(logits, target)
