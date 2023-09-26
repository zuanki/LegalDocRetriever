import torch
import torch.nn as nn
from transformers import BertModel

class BertCLS(nn.Module):
    def __init__(
        self,
        bert_model: BertModel
    ):
        super(BertCLS, self).__init__()
        self.bert_model = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

        # Freeze bert_model
        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # [CLS] token
        cls_embedding = outputs[0][:, 0, :]

        cls_embedding = self.dropout(cls_embedding)

        logits = self.classifier(cls_embedding)
        logits = self.softmax(logits)

        return logits
    
    def __str__(self) -> str:
        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())

        return super().__str__() + f"\nTrainable params: {num_trainable_params}\nTotal params: {total_params}\nTrainable percentage: {num_trainable_params / total_params * 100:.5f}%"