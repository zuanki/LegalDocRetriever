import torch
import torch.nn as nn

from transformers import BertModel

class BertAdapter(nn.Module):
    def __init__(self):
        super(BertAdapter, self).__init__()

    def forward(self):
        pass

    def __str__(self) -> str:
        n_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return super().__str__() + f"\nTrainable params: {n_trainable_params}"