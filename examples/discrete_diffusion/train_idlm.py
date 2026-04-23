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

"""
Reference training script for converting an AR causal LM into an I-DLM checkpoint.

I-DLM's training objective (Yu et al., 2026) is *all-masked* finetuning: the input is the concatenation of a
fully-masked copy `x_t` and the clean copy `x_0` under strict causal attention, with two CE losses:

    L = CE_noisy(predict clean tokens from masked positions) + alpha * CE_clean(next-token on clean region,
                                                                                with Dream shift)

Each forward is a single pass over `[prompt | MASK * gen_len | prompt | x_0_gen]` with strict causal masking.
Under the Dream shift, `logits[:, i, :]` predicts the token at input position `i+1`.

This script is intentionally minimal — enough to demonstrate the loss construction and integrate with
`accelerate`. Real I-DLM training requires a larger dataloader, gradient checkpointing, and packing logic; see
`3rd_party/I-DLM/training/` for the full LlamaFactory-based pipeline.

Example:
    accelerate launch train_idlm.py \\
      --model_name_or_path Qwen/Qwen3-8B \\
      --dataset_name HuggingFaceH4/ultrachat_200k \\
      --text_column prompt \\
      --output_dir /tmp/idlm-qwen3 \\
      --max_train_steps 500
"""

import argparse
import math
import os
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

from diffusers import IDLMBlockDiffusionScheduler


logger = get_logger(__name__)


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_name: str
    dataset_config_name: str | None
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
    warmup_steps: int

    max_seq_length: int
    prompt_length: int
    clean_loss_weight: float  # `alpha` in L = CE_noisy + alpha * CE_clean
    mask_token: str


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Minimal I-DLM all-masked training loop.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--dataset_config_name", type=str, default=None)
    p.add_argument("--text_column", type=str, default="text")

    p.add_argument("--output_dir", type=str, default="./idlm-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_steps", type=int, default=1000)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=10)

    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--warmup_steps", type=int, default=50)

    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--prompt_length", type=int, default=256)
    p.add_argument("--clean_loss_weight", type=float, default=1.0)
    p.add_argument("--mask_token", type=str, default="<|MASK|>")

    args = p.parse_args()
    return TrainConfig(**vars(args))


def build_tokenizer(cfg: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": cfg.mask_token})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_fn(examples, tokenizer, text_column: str, max_length: int):
    out = tokenizer(
        examples[text_column],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    return out


def collate(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
    }


def compute_idlm_loss(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    mask_token_id: int,
    prompt_length: int,
    clean_loss_weight: float,
    scheduler: IDLMBlockDiffusionScheduler,
):
    """
    Single-pass all-masked I-DLM loss.

    Constructs a fully-masked `x_t` by replacing tokens after `prompt_length` with `mask_token_id`, forwards
    through the model with standard causal attention, and computes:
      * `ce_noisy` — CE on the model's logits at masked positions against the *original* token
        (predicting the clean token from the masked input).
      * `ce_clean` — CE on the model's logits at clean (non-masked) positions against the *next* token
        (Dream shift: logits[:, i, :] predicts token at position i+1).
    """
    labels = input_ids.clone()
    noisy, _clean, noisy_mask = scheduler.add_noise(
        input_ids,
        attention_mask,
        prompt_length=prompt_length,
        mask_token_id=mask_token_id,
    )
    out = model(input_ids=noisy, attention_mask=attention_mask, use_cache=False)
    logits = out.logits  # [B, T, V]

    # Dream shift: logits[:, i, :] predicts token i+1. Align predictions with shifted labels.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_valid = attention_mask[:, 1:].to(dtype=torch.bool)
    shift_noisy = noisy_mask[:, 1:]  # label at pos i+1 was masked in the input
    shift_clean = (~shift_noisy) & shift_valid

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    per_token_ce = loss_fct(flat_logits.float(), flat_labels).view(shift_labels.shape)

    noisy_mask_flat = shift_noisy & shift_valid
    ce_noisy = (per_token_ce * noisy_mask_flat).sum() / noisy_mask_flat.sum().clamp_min(1)
    ce_clean = (per_token_ce * shift_clean).sum() / shift_clean.sum().clamp_min(1)

    loss = ce_noisy + clean_loss_weight * ce_clean
    return loss, {"ce_noisy": ce_noisy.detach(), "ce_clean": ce_clean.detach()}


def main():
    cfg = parse_args()

    accelerator_project_config = ProjectConfiguration(project_dir=cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        project_config=accelerator_project_config,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    tokenizer = build_tokenizer(cfg)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "Tokenizer must define a mask token."

    raw = load_dataset(cfg.dataset_name, cfg.dataset_config_name, split="train", streaming=False)
    raw = raw.map(
        lambda ex: tokenize_fn(ex, tokenizer, cfg.text_column, cfg.max_seq_length),
        batched=True,
        remove_columns=raw.column_names,
    )
    loader = DataLoader(
        raw.with_format("torch"),
        batch_size=cfg.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    num_update_steps = math.ceil(cfg.max_train_steps * cfg.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_update_steps,
    )
    diffusion_scheduler = IDLMBlockDiffusionScheduler()

    model, optimizer, loader, lr_scheduler = accelerator.prepare(model, optimizer, loader, lr_scheduler)

    global_step = 0
    model.train()
    while global_step < cfg.max_train_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                loss, extras = compute_idlm_loss(
                    model,
                    batch["input_ids"],
                    batch["attention_mask"],
                    mask_token_id=mask_token_id,
                    prompt_length=cfg.prompt_length,
                    clean_loss_weight=cfg.clean_loss_weight,
                    scheduler=diffusion_scheduler,
                )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg.logging_steps == 0:
                    logger.info(
                        f"step {global_step}  loss={loss.item():.4f}  "
                        f"ce_noisy={extras['ce_noisy'].item():.4f}  "
                        f"ce_clean={extras['ce_clean'].item():.4f}"
                    )
                if global_step % cfg.checkpointing_steps == 0 and accelerator.is_main_process:
                    save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    accelerator.unwrap_model(model).save_pretrained(save_dir)
                    tokenizer.save_pretrained(save_dir)
                if global_step >= cfg.max_train_steps:
                    break

    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)


if __name__ == "__main__":
    main()
