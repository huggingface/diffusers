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
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler


logger = get_logger(__name__)


@dataclass
class TrainConfig:
    draft_model_id: str
    target_model_id: str
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
    block_size: int
    mask_token: str


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Fine-tune a DFlash draft model with target-conditioned blocks.")

    parser.add_argument("--draft_model_id", type=str, default="z-lab/Qwen3-4B-DFlash-b16")
    parser.add_argument("--target_model_id", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--text_column", type=str, default="text")

    parser.add_argument("--output_dir", type=str, default="dflash-output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument(
        "--lr_scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts"]
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=100)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--block_size", type=int, default=0, help="Override draft block size (0 uses the model config)."
    )
    parser.add_argument("--mask_token", type=str, default="<|MASK|>")

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def tokenize_fn(examples: Dict, tokenizer, text_column: str, max_length: int):
    texts = examples[text_column]
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) > 0]
    return tokenizer(texts, truncation=True, padding=False, max_length=max_length)


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start = 1
    end = int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(int(num_draft_layers))]


def extract_context_feature(hidden_states, layer_ids):
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def get_target_input_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        base = getattr(model, "model", None)
        embeddings = getattr(base, "embed_tokens", None)
    if embeddings is None:
        raise ValueError("Target model must expose input embeddings.")
    return embeddings


def get_target_output_embeddings(model: torch.nn.Module) -> torch.nn.Module:
    embeddings = model.get_output_embeddings()
    if embeddings is None:
        embeddings = getattr(model, "lm_head", None)
    if embeddings is None:
        raise ValueError("Target model must expose output embeddings.")
    return embeddings


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

    tokenizer = AutoTokenizer.from_pretrained(cfg.target_model_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": cfg.mask_token})

    draft_model = AutoModel.from_pretrained(cfg.draft_model_id, trust_remote_code=True)
    target_model = AutoModelForCausalLM.from_pretrained(cfg.target_model_id)
    target_model.eval()
    target_model.requires_grad_(False)

    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError("Tokenizer must define a mask token for DFlash training.")

    input_embeddings = get_target_input_embeddings(target_model)
    output_embeddings = get_target_output_embeddings(target_model)

    block_size = int(cfg.block_size)
    if block_size <= 0:
        block_size = getattr(draft_model, "block_size", None) or getattr(
            getattr(draft_model, "config", None), "block_size", None
        )
    if block_size is None:
        raise ValueError("Draft model must define `block_size` or pass --block_size.")
    block_size = int(block_size)
    if block_size < 2:
        raise ValueError("`block_size` must be at least 2 for DFlash training.")

    layer_ids = getattr(draft_model, "target_layer_ids", None)
    if layer_ids is None:
        cfg_draft = getattr(draft_model, "config", None)
        num_target_layers = getattr(cfg_draft, "num_target_layers", None)
        num_hidden_layers = getattr(cfg_draft, "num_hidden_layers", None)
        if num_target_layers is None or num_hidden_layers is None:
            raise ValueError("Draft model must expose `target_layer_ids` or `num_target_layers` in config.")
        layer_ids = build_target_layer_ids(int(num_target_layers), int(num_hidden_layers))

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

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")
    train_dataloader = DataLoader(
        tokenized, shuffle=True, collate_fn=collator, batch_size=cfg.per_device_train_batch_size, drop_last=True
    )

    optimizer = torch.optim.AdamW(draft_model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=cfg.max_train_steps,
    )

    draft_model, optimizer, train_dataloader, lr_scheduler, target_model = accelerator.prepare(
        draft_model, optimizer, train_dataloader, lr_scheduler, target_model
    )
    input_embeddings = get_target_input_embeddings(target_model)
    output_embeddings = get_target_output_embeddings(target_model)

    global_step = 0
    draft_model.train()

    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            with accelerator.accumulate(draft_model):
                input_ids = batch["input_ids"]
                attention_mask = batch.get("attention_mask", torch.ones_like(input_ids))

                valid_lengths = attention_mask.sum(dim=1)
                min_valid = int(valid_lengths.min().item())
                if min_valid <= block_size:
                    continue

                max_start = min_valid - block_size
                start = torch.randint(1, max_start + 1, (1,), device=input_ids.device).item()

                block_output_ids = torch.full(
                    (input_ids.shape[0], block_size),
                    int(mask_token_id),
                    device=input_ids.device,
                    dtype=torch.long,
                )
                block_output_ids[:, 0] = input_ids[:, start]
                block_targets = input_ids[:, start + 1 : start + block_size]
                block_mask = attention_mask[:, start + 1 : start + block_size]

                position_ids = torch.arange(start, start + block_size, device=input_ids.device).unsqueeze(0)
                position_ids = position_ids.expand(input_ids.shape[0], -1)

                with torch.no_grad():
                    target_out = target_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    target_hidden = extract_context_feature(target_out.hidden_states, layer_ids)
                    target_hidden = target_hidden[:, :start, :]

                noise_embedding = input_embeddings(block_output_ids)
                draft_hidden = draft_model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids,
                    use_cache=False,
                    is_causal=False,
                )
                if not torch.is_tensor(draft_hidden):
                    draft_hidden = getattr(draft_hidden, "last_hidden_state", draft_hidden[0])

                logits = output_embeddings(draft_hidden[:, -block_size + 1 :, :])
                vocab_size = logits.shape[-1]
                loss = F.cross_entropy(logits.view(-1, vocab_size), block_targets.reshape(-1), reduction="none")
                loss = loss.view(block_targets.shape[0], -1)
                loss = (loss * block_mask.to(loss.dtype)).sum() / block_mask.sum().clamp_min(1)

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
                        unwrapped = accelerator.unwrap_model(draft_model)
                        unwrapped.save_pretrained(save_dir, save_function=accelerator.save)
                        tokenizer.save_pretrained(save_dir)

                if global_step >= cfg.max_train_steps:
                    break

        if global_step >= cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(draft_model)
        unwrapped.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
