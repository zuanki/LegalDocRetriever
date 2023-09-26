import pandas as pd
from transformers import BertTokenizer

import torch
from typing import List, Dict, Any
from torch.utils.data import Dataset


class LawDataset(Dataset):
    def __init__(
        self,
        path_df: str,
        tokenizer: BertTokenizer,
        max_len: int = 512,
        train: bool = True
    ):
        self.df = pd.read_csv(path_df)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train = train

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # [CLS] query [SEP] document [SEP]
        encoding = self.tokenizer.encode_plus(
            row['query'],
            row['document'],
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            truncation_strategy='only_second',  # truncate document
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )

        if self.train:
            # One-hot label
            label = torch.zeros(2)
            label[row['label']] = 1

            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'label': label.float()
            }

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
        }
