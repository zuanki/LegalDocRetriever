import os
import time
import shutil
import datetime
import argparse
import importlib
from tqdm import tqdm

import transformers
transformers.logging.set_verbosity_error()

from accelerate import Accelerator
from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='configs.base_config', help='config file path')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    module = importlib.import_module(args.config)
    config = module.Config()

    # Save
    date = datetime.date.today().strftime("%Y-%m-%d")
    _time = datetime.datetime.now().strftime("%H%M%S")

    time_str = f"{date}_{_time}"
    save_folder = f"{config.SAVE_DIR}/{time_str}"
    os.makedirs(f"{save_folder}/ckpts", exist_ok=True)

    # Write config
    shutil.copy(module.__file__, save_folder)

    # Init Accelerate
    accelerator = Accelerator(
        gradient_accumulation_steps=config.ACCUMULATE_STEPS
    )

    # Model
    model = config.model

    # Dataloader
    train_dataloader = config.train_dataloader

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
        logger.info(f"Epoch: {epoch} \t Loss: {epoch_loss}", main_process_only=True)
        if accelerator.is_main_process:
            with open(f"{save_folder}/exp.log", 'a') as f:
                f.write(f"Epoch: {epoch} \t Loss: {epoch_loss} \n")

        # Saving
        accelerator.wait_for_everyone()
        model_state_dict = accelerator.get_state_dict(model)
        accelerator.save(model_state_dict, f"{save_folder}/ckpts/{epoch}.pt")

    end_time = time.time()
    logger.info(f"Training time: {(end_time - start_time)/3600:.2f}", main_process_only=True)
    if accelerator.is_main_process:
        with open(f"{save_folder}/exp.log", 'a') as f:
            f.write(f"Training time: {(end_time - start_time)/3600:.2f}")

if __name__ == "__main__":
    main()