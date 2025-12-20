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
from typing import Dict, Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler

from diffusers.training_utils import compute_confidence_aware_loss


logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config_name: Optional[str]
    text_column: str
    cache_dir: Optional[str]
    use_dummy_data: bool
    num_dummy_samples: int

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
    prompt_length: int
    block_length: int

    lambda_conf: float
    conf_temperature: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train block-refinement with a confidence-aware loss on a causal LM.")

    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--use_dummy_data", action="store_true", help="Use random-token data instead of downloading.")
    parser.add_argument("--num_dummy_samples", type=int, default=2048)

    parser.add_argument("--output_dir", type=str, default="qwen-block-refinement-output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts"]
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--prompt_length", type=int, default=32)
    parser.add_argument("--block_length", type=int, default=32)

    parser.add_argument("--lambda_conf", type=float, default=2.0)
    parser.add_argument("--conf_temperature", type=float, default=0.5)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def tokenize_fn(examples: Dict, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    return tokenizer(texts, truncation=True, padding=False, max_length=max_length)


class RandomTokenDataset(torch.utils.data.Dataset):
    def __init__(self, *, num_samples: int, seq_len: int, vocab_size: int, pad_token_id: int):
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.pad_token_id = int(pad_token_id)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def forward_process_semi_ar(
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    prompt_length: int,
    block_length: int,
    mask_token_id: int,
    generator: Optional[torch.Generator],
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    noisy = input_ids.clone()
    noisy_rev = input_ids.clone()
    masked = torch.zeros_like(input_ids, dtype=torch.bool)
    masked_rev = torch.zeros_like(input_ids, dtype=torch.bool)

    # Only mask non-padding positions after the prompt.
    valid = attention_mask.to(dtype=torch.bool)
    start = int(prompt_length)
    for block_start in range(start, seq_len, int(block_length)):
        block_end = min(seq_len, block_start + int(block_length))
        seg_len = block_end - block_start
        if seg_len <= 0:
            continue

        p_mask = torch.rand((batch_size, 1), device=device, generator=generator)
        seg = torch.rand((batch_size, seg_len), device=device, generator=generator) < p_mask
        seg = seg & valid[:, block_start:block_end]
        seg_rev = (~seg) & valid[:, block_start:block_end]

        masked[:, block_start:block_end] = seg
        masked_rev[:, block_start:block_end] = seg_rev

    noisy = torch.where(masked, torch.full_like(noisy, int(mask_token_id)), noisy)
    noisy_rev = torch.where(masked_rev, torch.full_like(noisy_rev, int(mask_token_id)), noisy_rev)
    return noisy, noisy_rev, masked, masked_rev


def main():
    cfg = parse_args()
    if cfg.prompt_length >= cfg.max_length:
        raise ValueError("`prompt_length` must be < `max_length`.")
    if cfg.block_length <= 0:
        raise ValueError("`block_length` must be > 0.")

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

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, use_fast=True, cache_dir=cfg.cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})

    load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, cache_dir=cfg.cache_dir, torch_dtype=load_dtype
    )
    model.resize_token_embeddings(len(tokenizer))
    if load_dtype == torch.float32:
        model.to(dtype=torch.float32)

    mask_token_id = int(tokenizer.mask_token_id)

    if cfg.use_dummy_data:
        dataset = RandomTokenDataset(
            num_samples=cfg.num_dummy_samples,
            seq_len=cfg.max_length,
            vocab_size=len(tokenizer),
            pad_token_id=int(tokenizer.pad_token_id),
        )
        train_dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=cfg.per_device_train_batch_size,
            drop_last=True,
        )
    else:
        raw_datasets = load_dataset(cfg.dataset_name, cfg.dataset_config_name, cache_dir=cfg.cache_dir)
        if "train" not in raw_datasets:
            raise ValueError(f"Dataset {cfg.dataset_name} has no 'train' split.")

        with accelerator.main_process_first():
            tokenized = raw_datasets["train"].map(
                lambda ex: tokenize_fn(ex, tokenizer, cfg.text_column, cfg.max_length),
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Tokenizing",
            )

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

    for _epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))

                gen = torch.Generator(device=input_ids.device).manual_seed(cfg.seed + global_step)
                noisy, noisy_rev, masked, masked_rev = forward_process_semi_ar(
                    input_ids,
                    attention_mask,
                    prompt_length=int(cfg.prompt_length),
                    block_length=int(cfg.block_length),
                    mask_token_id=mask_token_id,
                    generator=gen,
                )

                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
                )

                logits = model(input_ids=noisy, attention_mask=attention_mask, position_ids=position_ids).logits
                logits_rev = model(
                    input_ids=noisy_rev, attention_mask=attention_mask, position_ids=position_ids
                ).logits

                logits = logits.clone()
                logits[..., mask_token_id] = torch.finfo(logits.dtype).min
                logits_rev = logits_rev.clone()
                logits_rev[..., mask_token_id] = torch.finfo(logits_rev.dtype).min

                valid = attention_mask.to(dtype=torch.bool)
                masked = masked & valid
                masked_rev = masked_rev & valid

                labels = input_ids.clone()
                labels[~masked] = -100
                labels_rev = input_ids.clone()
                labels_rev[~masked_rev] = -100

                weights = masked.to(dtype=logits.dtype)
                weights_rev = masked_rev.to(dtype=logits.dtype)

                loss, loss_sft, loss_conf = compute_confidence_aware_loss(
                    logits,
                    labels,
                    lambda_conf=cfg.lambda_conf,
                    temperature=cfg.conf_temperature,
                    per_token_weights=weights,
                )
                loss_rev, loss_sft_rev, loss_conf_rev = compute_confidence_aware_loss(
                    logits_rev,
                    labels_rev,
                    lambda_conf=cfg.lambda_conf,
                    temperature=cfg.conf_temperature,
                    per_token_weights=weights_rev,
                )

                total_loss = loss + loss_rev
                accelerator.backward(total_loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    logger.info(
                        "step=%d loss=%.4f sft=%.4f conf=%.4f lr=%.6g",
                        global_step,
                        total_loss.item(),
                        (loss_sft + loss_sft_rev).item(),
                        (loss_conf + loss_conf_rev).item(),
                        lr_scheduler.get_last_lr()[0],
                    )
                    print(
                        f"step={global_step} loss={total_loss.item():.4f} "
                        f"sft={(loss_sft + loss_sft_rev).item():.4f} "
                        f"conf={(loss_conf + loss_conf_rev).item():.4f} "
                        f"lr={lr_scheduler.get_last_lr()[0]:.6g}"
                    )

                if cfg.checkpointing_steps > 0 and global_step % cfg.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        accelerator.unwrap_model(model).save_pretrained(save_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(save_dir)

                if global_step >= cfg.max_train_steps:
                    break

        if global_step >= cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
