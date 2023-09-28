from torch.utils.data import DataLoader
from src.datasets import LawDataset
from src.utils import print_trainable_parameters
from accelerate import Accelerator
from accelerate.logging import get_logger
import os
import time
import shutil
import datetime
import argparse
import importlib
from tqdm import tqdm

import torch
import transformers
transformers.logging.set_verbosity_error()


logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str,
                        default='configs.base_config', help='config file path')
    parser.add_argument('--data_path', type=str,
                        default='data/BM25/2022/train.csv', help='data path')
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints/cls', help='save dir')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    TRAIN_DATA_PATH = args.data_path
    SAVE_DIR = args.save_dir

    module = importlib.import_module(args.config)
    config = module.Config()

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    _time = datetime.datetime.now().strftime("%H%M%S")

    time_str = f"{date}_{_time}"
    save_folder = f"{SAVE_DIR}/{time_str}"
    os.makedirs(f"{save_folder}/ckpts", exist_ok=True)

    # Write config
    shutil.copy(module.__file__, save_folder)

    # Init Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_STEPS
    )

    # Model
    model = config.model
    print_trainable_parameters(model)

    # Dataloader
    # train_dataloader = config.train_dataloader
    train_dataset = LawDataset(
        path_df=TRAIN_DATA_PATH,
        tokenizer=config.tokenizer,
        max_len=512,
        train=True
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True
    )

    # Loss
    loss_fn = config.LOSS_FN

    # Optimizer and Scheduler
    optimizer = config.OPTIMIZER(
        model.parameters(),
        **config.OPTIMIZER_KWARGS
    )
    scheduler = config.SCHEDULER(
        optimizer,
        **config.SCHEDULER_KWARGS
    )

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    # Training Loop

    start_time = time.time()

    for epoch in range(1, config.MAX_EPOCHS + 1):
        epoch_losses = []

        for batch in tqdm(train_dataloader, desc=f"epoch: {epoch}", disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]

                label = batch["label"]

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )

                loss = loss_fn(logits, label)

                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                epoch_losses.append(loss.item())

        # After each epoch
        scheduler.step()
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}",
                    main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

        # Saving
        if config.LORA:
            if (epoch + 1) % config.SAVE_EVERY == 0:
                # TODO:
                # accelerator.wait_for_everyone()
                os.makedirs(
                    f"{save_folder}/ckpts/lora_{epoch}", exist_ok=True)
                # Save Lora
                model.save_pretrained(
                    f"{save_folder}/ckpts/lora_{epoch}")

                # Save classifier weights
                classifier_state_dict = model.base_model.classifier.state_dict()
                torch.save(
                    classifier_state_dict, f"{save_folder}/ckpts/lora_{epoch}/classifier.pt")
        else:
            if (epoch + 1) % config.SAVE_EVERY == 0:
                accelerator.wait_for_everyone()
                model_state_dict = accelerator.get_state_dict(model)
                accelerator.save(model_state_dict,
                                 f"{save_folder}/ckpts/{epoch}.pt")

    # Save epoch MAX + 1
    if config.LORA:
        # TODO:
        # accelerator.wait_for_everyone()
        os.makedirs(
            f"{save_folder}/ckpts/lora_{config.MAX_EPOCHS + 1}", exist_ok=True)
        # Save Lora
        model.save_pretrained(
            f"{save_folder}/ckpts/lora_{config.MAX_EPOCHS + 1}")
        # Save classifier weights
        classifier_state_dict = model.base_model.classifier.state_dict()
        torch.save(
            classifier_state_dict, f"{save_folder}/ckpts/lora_{config.MAX_EPOCHS + 1}/classifier.pt")
    else:
        accelerator.wait_for_everyone()
        model_state_dict = accelerator.get_state_dict(model)
        accelerator.save(model_state_dict,
                         f"{save_folder}/ckpts/{config.MAX_EPOCHS + 1}.pt")

    end_time = time.time()
    logger.info(
        f"Training time: {(end_time - start_time)/3600:.2f}", main_process_only=True)
    if accelerator.is_main_process:
        with open(f"{save_folder}/exp.log", 'a') as f:
            f.write(f"Training time: {(end_time - start_time)/3600:.2f}")


if __name__ == "__main__":
    main()

# accelerate launch --mixed_precision fp16 train.py configs.base_config --data_path data/BM25/2022/train.csv --save_dir checkpoints/cls
