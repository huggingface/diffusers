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

I-DLM's training objective (Yu et al., 2026) is *all-masked* finetuning. Each forward concatenates a
fully-masked copy `x_t` and the clean copy `x_0` into a length-`2L` sequence run under the block-diffusion
attention mask (causal within the noisy blocks, cross-attention from `x_t` to the clean prefix, strict-causal
`x_0`), with a Dream next-token shift (`logits[:, i, :]` predicts token `i+1`). Two CE losses, both on the
response tokens:

    L = CE_noisy(decode q: masked tokens conditioned on the clean prefix)
        + alpha * CE_clean(verify p: clean response copy, strict causal)

`block_length` is the diffusion block size; the paper trains a b1 -> b2 -> b3 curriculum (one epoch each),
enabling `--auto_balance` (alpha = CE_noisy / CE_clean, detached) at the b3 stage.

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
    block_length: int  # diffusion block size (paper curriculum b1 -> b2 -> b3)
    clean_loss_weight: float  # `alpha` in L = CE_noisy + alpha * CE_clean
    auto_balance: bool  # replace alpha with (CE_noisy / CE_clean).detach() (paper Eq. 2, b3)
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
    p.add_argument("--block_length", type=int, default=1)
    p.add_argument("--clean_loss_weight", type=float, default=0.2)
    p.add_argument("--auto_balance", action="store_true")
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


def build_block_diffusion_mask(
    seq_len: int,
    block_size: int,
    valid_mask: torch.LongTensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive ``[x_t | x_0]`` block-diffusion attention mask (paper Appendix E).

    Built over the concatenated length-``2L`` sequence (noisy copy then clean copy):
      * ``M_BD``  — causal self-attention within each noisy ``x_t`` block,
      * ``M_OBC`` — each ``x_t`` token cross-attends clean ``x_0`` tokens in strictly
        earlier blocks (the clean ground-truth prefix the decode is conditioned on),
      * ``M_BC``  — strict token-causal attention within the clean ``x_0`` copy.

    Returns a ``[B, 1, 2L, 2L]`` float mask (``0`` attend, ``-inf`` block).
    """
    q = torch.arange(2 * seq_len, device=device).view(1, 1, -1, 1)
    kv = torch.arange(2 * seq_len, device=device).view(1, 1, 1, -1)
    x0_q, x0_kv = q >= seq_len, kv >= seq_len
    block_q = torch.where(x0_q, (q - seq_len) // block_size, q // block_size)
    block_kv = torch.where(x0_kv, (kv - seq_len) // block_size, kv // block_size)

    m_bd = (block_q == block_kv) & (~x0_q) & (~x0_kv) & (q >= kv)
    m_obc = (block_q > block_kv) & x0_kv & (~x0_q)
    m_bc = (q >= kv) & x0_q & x0_kv

    key_valid = torch.cat([valid_mask.bool(), valid_mask.bool()], dim=1).view(valid_mask.size(0), 1, 1, 2 * seq_len)
    allow = (m_bd | m_obc | m_bc) & key_valid
    return torch.where(allow, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(float("-inf"), device=device, dtype=dtype))


def compute_idlm_loss(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    *,
    mask_token_id: int,
    prompt_length: int,
    block_length: int,
    clean_loss_weight: float,
    auto_balance: bool,
    scheduler: IDLMBlockDiffusionScheduler,
):
    """
    I-DLM block-diffusion loss (matches the official trainer).

    Concatenates the fully-masked copy ``x_t`` and the clean copy ``x_0`` into a length-``2L``
    sequence, runs one forward under the block-diffusion attention mask, and applies a Dream
    next-token shift. Both cross-entropy terms are supervised on the response tokens:
      * ``ce_noisy`` — decode CE on the ``x_t`` half (each masked token conditioned on the clean prefix),
      * ``ce_clean`` — verify CE on the ``x_0`` half (clean response copy, strict causal).
    """
    labels = input_ids.clone()
    noisy, _clean, noisy_mask = scheduler.add_noise(
        input_ids,
        attention_mask,
        prompt_length=prompt_length,
        mask_token_id=mask_token_id,
    )
    bsz, L = input_ids.shape
    device = input_ids.device

    concat_ids = torch.cat([noisy, input_ids], dim=1)  # [x_t | x_0], length 2L
    pos = torch.arange(L, device=device)
    concat_pos = torch.cat([pos, pos]).unsqueeze(0).expand(bsz, -1)
    block_mask = build_block_diffusion_mask(
        L, block_length, attention_mask, dtype=next(model.parameters()).dtype, device=device
    )

    out = model(input_ids=concat_ids, attention_mask=block_mask, position_ids=concat_pos, use_cache=False)
    logits = out.logits  # [B, 2L, V]
    noisy_logits, clean_logits = logits[:, :L, :], logits[:, L : 2 * L, :]

    # Dream shift: logits[:, i] predicts token i+1. Both copies supervise the response.
    shift_target = labels[:, 1:]
    supervise = noisy_mask[:, 1:].bool() & attention_mask[:, 1:].bool()
    weight = supervise.float()
    denom = weight.sum().clamp_min(1)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    def ce(half):
        per_token = loss_fct(
            half[:, :-1, :].reshape(-1, half.size(-1)).float(), shift_target.reshape(-1)
        ).view(shift_target.shape)
        return (per_token * weight).sum() / denom

    ce_noisy, ce_clean = ce(noisy_logits), ce(clean_logits)
    alpha = (ce_noisy / ce_clean.clamp_min(1e-6)).detach() if auto_balance else clean_loss_weight
    loss = ce_noisy + alpha * ce_clean
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
    # sdpa honours the 4D block-diffusion mask; FlashAttention-2 would ignore it.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, trust_remote_code=True, attn_implementation="sdpa"
    )
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
                    block_length=cfg.block_length,
                    clean_loss_weight=cfg.clean_loss_weight,
                    auto_balance=cfg.auto_balance,
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
