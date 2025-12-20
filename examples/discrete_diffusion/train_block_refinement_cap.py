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
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import DataLoader, Dataset

from diffusers import BlockRefinementPipeline
from diffusers.training_utils import compute_confidence_aware_loss


logger = get_logger(__name__)


@dataclass
class TrainConfig:
    output_dir: str
    seed: int
    max_train_steps: int
    logging_steps: int
    checkpointing_steps: int

    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float

    vocab_size: int
    mask_token_id: int
    eos_token_id: int
    max_length: int
    prompt_length: int

    block_length: int
    steps: int
    lambda_conf: float
    conf_temperature: float
    temperature: float
    threshold: float


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train a block-wise refinement model with a confidence-aware objective (CAP-style)."
    )

    parser.add_argument("--output_dir", type=str, default="block-refinement-output")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--checkpointing_steps", type=int, default=500)

    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--mask_token_id", type=int, default=255)
    parser.add_argument("--eos_token_id", type=int, default=254)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--prompt_length", type=int, default=8)

    parser.add_argument("--block_length", type=int, default=16)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--lambda_conf", type=float, default=2.0)
    parser.add_argument("--conf_temperature", type=float, default=0.5)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.95)

    args = parser.parse_args()
    return TrainConfig(**vars(args))


def build_block_attention_mask(
    *,
    num_blocks: int,
    block_length: int,
    total_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=device, dtype=torch.bool))
    attn = (
        block_mask.repeat_interleave(block_length, dim=0)
        .repeat_interleave(block_length, dim=1)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    attn = attn[:, :, :total_length, :total_length]
    return torch.where(
        attn, torch.zeros((), device=device, dtype=dtype), torch.full((), float("-inf"), device=device, dtype=dtype)
    )


def forward_process_semi_ar(
    input_ids: torch.LongTensor,
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

    start = int(prompt_length)
    for block_start in range(start, seq_len, int(block_length)):
        block_end = min(seq_len, block_start + int(block_length))
        seg_len = block_end - block_start
        if seg_len <= 0:
            continue

        p_mask = torch.rand((batch_size, 1), device=device, generator=generator)
        seg = torch.rand((batch_size, seg_len), device=device, generator=generator) < p_mask
        seg_rev = ~seg

        masked[:, block_start:block_end] = seg
        masked_rev[:, block_start:block_end] = seg_rev

    noisy = torch.where(masked, torch.full_like(noisy, int(mask_token_id)), noisy)
    noisy_rev = torch.where(masked_rev, torch.full_like(noisy_rev, int(mask_token_id)), noisy_rev)
    return noisy, noisy_rev, masked, masked_rev


class RandomTokenDataset(Dataset):
    def __init__(self, *, num_samples: int, seq_len: int, vocab_size: int, eos_token_id: int):
        self.num_samples = int(num_samples)
        self.seq_len = int(seq_len)
        self.vocab_size = int(vocab_size)
        self.eos_token_id = int(eos_token_id)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        del idx
        # Keep EOS out of the training distribution to avoid trivial early-stops during sampling.
        ids = torch.randint(0, self.vocab_size - 2, (self.seq_len,), dtype=torch.long)
        return {"input_ids": ids}


class TinyBlockRefinementLM(torch.nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int = 128, num_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)

        self.token_emb = torch.nn.Embedding(self.vocab_size, self.hidden_size)
        self.pos_emb = torch.nn.Embedding(2048, self.hidden_size)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_size * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.lm_head = torch.nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(self, input_ids, attention_mask=None, position_ids=None, **kwargs):
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        x = self.token_emb(input_ids) + self.pos_emb(position_ids)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 4:
                attn_mask = attention_mask[0, 0]
            elif attention_mask.ndim == 2:
                attn_mask = attention_mask
            else:
                raise ValueError(f"Unsupported `attention_mask` shape: {attention_mask.shape}")
            attn_mask = attn_mask.to(dtype=torch.float32)

        hidden = self.encoder(x, mask=attn_mask)
        logits = self.lm_head(hidden)
        return type("Output", (), {"logits": logits})


def save_checkpoint(output_dir: str, *, model: torch.nn.Module, cfg: TrainConfig):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True)


def main():
    cfg = parse_args()
    if cfg.mask_token_id >= cfg.vocab_size:
        raise ValueError("`mask_token_id` must be < `vocab_size`.")
    if cfg.eos_token_id >= cfg.vocab_size:
        raise ValueError("`eos_token_id` must be < `vocab_size`.")
    if cfg.prompt_length >= cfg.max_length:
        raise ValueError("`prompt_length` must be < `max_length`.")

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

    dataset = RandomTokenDataset(
        num_samples=max(cfg.max_train_steps * cfg.per_device_train_batch_size, 4096),
        seq_len=cfg.max_length,
        vocab_size=cfg.vocab_size,
        eos_token_id=cfg.eos_token_id,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.per_device_train_batch_size, shuffle=True, drop_last=True)

    model = TinyBlockRefinementLM(vocab_size=cfg.vocab_size)
    pipe = BlockRefinementPipeline(model=model, tokenizer=None)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    num_update_steps_per_epoch = math.ceil(len(dataloader) / cfg.gradient_accumulation_steps)
    num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    pipe = pipe.to(accelerator.device)

    global_step = 0
    model.train()

    for _epoch in range(num_train_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]

                # Build the same attention mask that the sampler uses.
                prompt_len = int(cfg.prompt_length)
                num_blocks = (prompt_len + int(cfg.max_length - prompt_len) + int(cfg.block_length) - 1) // int(
                    cfg.block_length
                )
                total_length = int(num_blocks) * int(cfg.block_length)
                total_length = max(total_length, int(cfg.max_length))
                attn_mask = build_block_attention_mask(
                    num_blocks=(total_length + int(cfg.block_length) - 1) // int(cfg.block_length),
                    block_length=int(cfg.block_length),
                    total_length=int(cfg.max_length),
                    device=input_ids.device,
                    dtype=torch.bfloat16 if input_ids.device.type == "cuda" else torch.float32,
                )
                position_ids = (
                    torch.arange(int(cfg.max_length), device=input_ids.device, dtype=torch.long)
                    .unsqueeze(0)
                    .expand_as(input_ids)
                )

                gen = None
                if accelerator.is_local_main_process:
                    gen = torch.Generator(device=input_ids.device).manual_seed(cfg.seed + global_step)

                noisy, noisy_rev, masked, masked_rev = forward_process_semi_ar(
                    input_ids,
                    prompt_length=prompt_len,
                    block_length=int(cfg.block_length),
                    mask_token_id=int(cfg.mask_token_id),
                    generator=gen,
                )

                logits = model(noisy, attention_mask=attn_mask, position_ids=position_ids).logits
                logits_rev = model(noisy_rev, attention_mask=attn_mask, position_ids=position_ids).logits

                # Do not allow predicting mask_id.
                logits = logits.clone()
                logits[..., int(cfg.mask_token_id)] = torch.finfo(logits.dtype).min
                logits_rev = logits_rev.clone()
                logits_rev[..., int(cfg.mask_token_id)] = torch.finfo(logits_rev.dtype).min

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
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1

                if global_step % cfg.logging_steps == 0 and accelerator.is_main_process:
                    logger.info(
                        "step=%d loss=%.4f sft=%.4f conf=%.4f",
                        global_step,
                        total_loss.item(),
                        (loss_sft + loss_sft_rev).item(),
                        (loss_conf + loss_conf_rev).item(),
                    )

                if cfg.checkpointing_steps > 0 and global_step % cfg.checkpointing_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_dir = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                        save_checkpoint(save_dir, model=accelerator.unwrap_model(model), cfg=cfg)

                if global_step >= cfg.max_train_steps:
                    break

        if global_step >= cfg.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(cfg.output_dir, "final")
        save_checkpoint(final_dir, model=accelerator.unwrap_model(model), cfg=cfg)

        # Quick sampler smoke to ensure the pipeline runs with the trained weights.
        out = pipe(
            prompt_ids=torch.randint(0, cfg.vocab_size - 2, (1, cfg.prompt_length), device=accelerator.device),
            gen_length=int(cfg.max_length - cfg.prompt_length),
            block_length=int(cfg.block_length),
            steps=int(cfg.steps),
            temperature=float(cfg.temperature),
            threshold=float(cfg.threshold),
            eos_early_stop=False,
            eos_token_id=int(cfg.eos_token_id),
            mask_token_id=int(cfg.mask_token_id),
            return_text=False,
        )
        logger.info("sample shape=%s", tuple(out.sequences.shape))

    logger.info("Done.")


if __name__ == "__main__":
    main()
