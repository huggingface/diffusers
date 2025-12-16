#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler,
)

from diffusers import TokenDiffusionScheduler


logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config_name: Optional[str]
    text_column: str

    output_dir: str
    seed: int
    max_train_steps: int
    checkpointing_steps: int
    logging_steps: int

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    lr_scheduler: str
    lr_warmup_steps: int

    max_length: int
    num_train_timesteps: int
    alpha_schedule: str
    eps: float
    sigma_min: float
    sigma_max: float
    min_timestep: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train an absorbing token diffusion LM (MDLM-style) with Accelerate.")

    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_column", type=str, default="text")

    parser.add_argument("--output_dir", type=str, default="mdlm-output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts"]
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument(
        "--alpha_schedule",
        type=str,
        default="log_linear",
        choices=["log_linear", "linear", "cosine", "geometric"],
    )
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--sigma_min", type=float, default=1e-4)
    parser.add_argument("--sigma_max", type=float, default=20.0)
    parser.add_argument("--min_timestep", type=int, default=1, help="Avoid t=0 to prevent 1/t weighting blow-ups.")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def tokenize_fn(examples: Dict, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    # drop empty lines
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    return tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_special_tokens_mask=True,
    )


def main():
    cfg = parse_args()

    project_config = ProjectConfiguration(project_dir=cfg.output_dir, logging_dir=os.path.join(cfg.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        os.makedirs(cfg.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    set_seed(cfg.seed)
    logger.info("Training configuration: %s", asdict(cfg))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True)
    if tokenizer.mask_token_id is None:
        # MDLM-style absorbing diffusion assumes a mask token exists.
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    config = AutoConfig.from_pretrained(cfg.model_name_or_path)
    model = AutoModelForMaskedLM.from_pretrained(cfg.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))

    scheduler = TokenDiffusionScheduler(
        vocab_size=len(tokenizer),
        mask_token_id=int(tokenizer.mask_token_id),
        num_train_timesteps=cfg.num_train_timesteps,
        alpha_schedule=cfg.alpha_schedule,
        eps=cfg.eps,
        sigma_min=cfg.sigma_min,
        sigma_max=cfg.sigma_max,
    )

    raw_datasets = load_dataset(cfg.dataset_name, cfg.dataset_config_name)
    if "train" not in raw_datasets:
        raise ValueError(f"Dataset {cfg.dataset_name} has no 'train' split.")

    with accelerator.main_process_first():
        tokenized = raw_datasets["train"].map(
            lambda ex: tokenize_fn(ex, tokenizer, cfg.text_column, cfg.max_length),
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Tokenizing",
        )

    # We reuse the standard MLM collator to pad and build attention masks; we won't use its masking.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    train_dataloader = DataLoader(
        tokenized, shuffle=True, collate_fn=collator, batch_size=cfg.per_device_train_batch_size, drop_last=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    model.train()

    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))

                # Sample discrete time indices (avoid timestep 0 for stability with 1/t weighting).
                min_t = max(1, int(cfg.min_timestep))
                max_t = scheduler.num_train_timesteps - 1
                timesteps = torch.randint(min_t, max_t + 1, (input_ids.shape[0],), device=input_ids.device)

                # Forward process q(x_t | x_0): replace tokens with [MASK] according to alpha(t).
                x_t = scheduler.add_noise(input_ids, noise=None, timesteps=timesteps)

                # Model predicts token logits for x0 reconstruction.
                logits = model(input_ids=x_t, attention_mask=attention_mask).logits  # [B, L, V]

                # MDLM-style constraints:
                # - Do not predict the mask token as x0.
                logits = logits.clone()
                logits[..., scheduler.mask_token_id] = torch.finfo(logits.dtype).min

                # Only compute loss on tokens that were masked by the forward process.
                mask_positions = x_t.eq(scheduler.mask_token_id) & attention_mask.to(dtype=torch.bool)
                per_token_ce = F.cross_entropy(logits.view(-1, logits.shape[-1]), input_ids.view(-1), reduction="none")
                per_token_ce = per_token_ce.view_as(input_ids)

                weights = scheduler.get_mdlm_loss_weights(timesteps)

                loss = (per_token_ce * mask_positions.to(per_token_ce.dtype) * weights).sum()
                denom = mask_positions.sum().clamp_min(1)
                loss = loss / denom

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    logger.info("step=%d loss=%.4f lr=%.6g", global_step, loss.item(), lr_scheduler.get_last_lr()[0])

                if cfg.checkpointing_steps > 0 and global_step % cfg.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(save_dir)
                        scheduler.save_pretrained(save_dir)

                if global_step >= cfg.max_train_steps:
                    break

        if global_step >= cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        scheduler.save_pretrained(final_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
