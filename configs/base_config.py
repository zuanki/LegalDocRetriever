import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer

from src.models import BertCLS
from src.datasets import LawDataset

from peft import LoraConfig, PeftModel, get_peft_model


class Config:
    def __init__(self):
        # BERT
        BERT_PRETRAINED_PATH = '/home/zuanki/Project/LegalDocRetriever/checkpoints/m_bert'
        bert_model: BertModel = BertModel.from_pretrained(BERT_PRETRAINED_PATH)
        bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            BERT_PRETRAINED_PATH)

        # Model
        self.model = BertCLS(bert_model)

        # Lora
        self.LORA = True
        LORA_CONFIG = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=8,
            bias="none",
            target_modules=["key", "value"],
            modules_to_save=["classifier"]
        )

        if self.LORA:
            self.model = get_peft_model(self.model, LORA_CONFIG)

        else:
            # Freeze bert_model
            for param in self.model.bert_model.parameters():
                param.requires_grad = False

        # DATA
        TRAIN_DATA_PATH: str = "/home/zuanki/Project/LegalDocRetriever/data/BM25/2022/train.csv"

        # Dataset and DataLoader
        self.train_dataset = LawDataset(
            path_df=TRAIN_DATA_PATH,
            tokenizer=bert_tokenizer,
            max_len=512,
            train=True
        )

        self.BATCH_SIZE: int = 8
        self.NUM_WORKERS: int = 0

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            shuffle=True
        )

        # Training
        self.MAX_EPOCHS: int = 10

        # Optimizer
        self.LEARNING_RATE: float = 1e-4
        self.ACCUMULATE_STEPS: int = 1

        self.OPTIMIZER = torch.optim.AdamW
        self.OPTIMIZER_KWARGS = {
            "lr": self.LEARNING_RATE,
            "weight_decay": 1e-2
        }

        self.SCHEDULER = torch.optim.lr_scheduler.CosineAnnealingLR
        self.SCHEDULER_KWARGS = {
            "T_max": self.MAX_EPOCHS
        }

        # Loss
        self.LOSS_FN = nn.BCEWithLogitsLoss()

        # Save
        self.SAVE_DIR = "/home/zuanki/Project/LegalDocRetriever/checkpoints/cls"
        self.SAVE_EVERY = 5
