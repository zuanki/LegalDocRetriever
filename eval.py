import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
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
                        default='checkpoints/cls/2023-09-28_100605/ckpts/lora_1', help='checkpoint path')

    args = parser.parse_args()

    return args


@torch.no_grad()
def evaluate():
    args = parse_args()
    DATA_PATH = args.data_path
    CHECKPOINT_PATH = args.checkpoint
    BATCH_SIZE = 1

    BERT_PRETRAINED_PATH = 'checkpoints/m_bert'
    bert_model: BertModel = BertModel.from_pretrained(BERT_PRETRAINED_PATH)
    bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
        BERT_PRETRAINED_PATH)

    model: BertCLS = BertCLS(bert_model)
    model = PeftModel.from_pretrained(model, CHECKPOINT_PATH)

    # Load classifier weights
    model.base_model.classifier.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_PATH, 'classifier.pt')))

    print_trainable_parameters(model)

    val_dataset = LawDataset(
        path_df=DATA_PATH,
        tokenizer=bert_tokenizer,
        max_len=512,
        train=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    tmp = next(iter(val_dataloader))

    print(model(
        input_ids=tmp['input_ids'],
        attention_mask=tmp['attention_mask'],
        token_type_ids=tmp['token_type_ids'],
    ))


if __name__ == "__main__":
    evaluate()
