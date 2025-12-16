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
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from diffusers import TokenDiffusionScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from an absorbing token diffusion LM (MDLM-style).")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path saved by train_mdlm.py (or compatible)."
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inject_bos", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def sample(
    model,
    tokenizer,
    scheduler: TokenDiffusionScheduler,
    *,
    num_samples: int,
    seq_len: int,
    num_inference_steps: int,
    generator: Optional[torch.Generator],
    inject_bos: bool,
    device: torch.device,
):
    scheduler.set_timesteps(num_inference_steps, device=device)

    x = torch.full((num_samples, seq_len), scheduler.mask_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(x, dtype=torch.long)

    if inject_bos and tokenizer.bos_token_id is not None:
        x[:, 0] = int(tokenizer.bos_token_id)

    for t in scheduler.timesteps:
        logits = model(input_ids=x, attention_mask=attention_mask).logits  # [B, L, V]
        x = scheduler.step(logits, t, x, generator=generator, return_dict=True).prev_sample

        if inject_bos and tokenizer.bos_token_id is not None:
            x[:, 0] = int(tokenizer.bos_token_id)

    return x


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path).to(device)
    scheduler = TokenDiffusionScheduler.from_pretrained(args.checkpoint_path)

    model.eval()

    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    samples = sample(
        model,
        tokenizer,
        scheduler,
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        num_inference_steps=args.num_inference_steps,
        generator=gen,
        inject_bos=args.inject_bos,
        device=device,
    )

    texts = tokenizer.batch_decode(samples, skip_special_tokens=True)
    for i, t in enumerate(texts):
        print(f"[{i}] {t}")


if __name__ == "__main__":
    main()
