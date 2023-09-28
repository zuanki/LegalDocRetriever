import os
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel
from transformers import BertModel, BertTokenizer

from src.models import BertCLS
from src.datasets import LawDataset
from src.utils import print_trainable_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,
                        default='data/BM25/2022/train.csv', help='data path')
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/cls/Kaggle/lora_20', help='checkpoint path')
    parser.add_argument('--result_path', type=str,
                        default='reports', help='checkpoint path')

    args = parser.parse_args()

    return args


@torch.no_grad()
def evaluate():
    args = parse_args()
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint
    RESULT_PATH = args.result_path
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BERT_PRETRAINED_PATH = 'checkpoints/m_bert'
    bert_model: BertModel = BertModel.from_pretrained(BERT_PRETRAINED_PATH)
    bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        BERT_PRETRAINED_PATH)

    model: BertCLS = BertCLS(bert_model)
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)

    # Load classifier weights
    model.base_model.classifier.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_PATH, 'classifier.pt'), map_location=DEVICE))

    model = model.to(DEVICE)
    print_trainable_parameters(model)

    val_dataset = LawDataset(
        path_df=DATA_PATH,
        tokenizer=bert_tokenizer,
        max_len=512,
        train=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
    )

    model.eval()
    y_pred = []

    for batch in tqdm(val_dataloader):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        token_type_ids = batch['token_type_ids'].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (batch_size, 2)

        y_pred.extend(logits[:, 1].cpu().numpy())

    df = pd.read_csv(DATA_PATH)
    df['score'] = y_pred

    df.to_csv(os.path.join(RESULT_PATH, 'result.csv'), index=False)


if __name__ == "__main__":
    evaluate()
