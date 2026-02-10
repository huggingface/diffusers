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
Sample script for MDLM-style absorbing token diffusion text generation.

This script demonstrates how to use the TokenDiffusionPipeline for unconditional
text generation using absorbing-state discrete diffusion.

Example usage:
    python sample_mdlm.py --model_id kuleshov-group/mdlm-owt --num_samples 4 --seq_len 64
"""

import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from diffusers import TokenDiffusionPipeline, TokenDiffusionScheduler


def main():
    parser = argparse.ArgumentParser(description="Sample from an absorbing token diffusion LM (MDLM-style).")
    parser.add_argument(
        "--model_id",
        type=str,
        default="kuleshov-group/mdlm-owt",
        help="HuggingFace model ID or path to local checkpoint.",
    )
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inject_bos", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for model/tokenizer.")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_id, trust_remote_code=args.trust_remote_code
    ).to(device)
    model.eval()

    mask_token_id = len(tokenizer)  # MDLM appends mask token after vocab
    vocab_size = mask_token_id + 1
    scheduler = TokenDiffusionScheduler(vocab_size=vocab_size, mask_token_id=mask_token_id)

    pipe = TokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"Generating {args.num_samples} samples of {args.seq_len} tokens with {args.num_inference_steps} steps")
    print("-" * 50)

    output = pipe(
        batch_size=args.num_samples,
        seq_len=args.seq_len,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        inject_start_token=args.inject_bos,
    )

    for i, text in enumerate(output.texts):
        print(f"[{i}] {text}")


if __name__ == "__main__":
    main()
