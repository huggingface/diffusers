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
Sample script for BD3LM-style block discrete diffusion text generation.

Example usage:
    python sample_bd3lm.py \
      --model_id kuleshov-group/bd3lm-owt-block_size4 \
      --revision refs/pr/2 \
      --gen_length 128 \
      --num_inference_steps 256 \
      --nucleus_p 0.9
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForMaskedLM, GPT2TokenizerFast

from diffusers import BD3LMPipeline, BD3LMTokenDiffusionScheduler


def main():
    parser = argparse.ArgumentParser(description="Generate text with BD3LM block diffusion.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="kuleshov-group/bd3lm-owt-block_size4",
        help="HuggingFace model ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch, tag, or commit hash).",
    )
    parser.add_argument("--gen_length", type=int, default=128, help="Number of tokens to generate.")
    parser.add_argument("--num_inference_steps", type=int, default=256, help="DDPM denoising steps per block.")
    parser.add_argument("--nucleus_p", type=float, default=0.9, help="Nucleus sampling threshold (1.0 to disable).")
    parser.add_argument("--noise_type", type=str, default="loglinear", help="Noise schedule type.")
    parser.add_argument("--eos_early_stop", action="store_true", help="Stop on EOS token.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model_id}")
    load_kwargs = {"trust_remote_code": True, "dtype": torch_dtype}
    if args.revision:
        load_kwargs["revision"] = args.revision

    config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True, revision=args.revision)
    config.attn_backend = "sdpa"  # Required for BD3LM sampling

    model = AutoModelForMaskedLM.from_pretrained(args.model_id, config=config, **load_kwargs).to(args.device)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # BD3LM convention: mask_token_id = vocab_size - 1
    mask_token_id = config.vocab_size - 1

    scheduler = BD3LMTokenDiffusionScheduler(
        block_size=config.block_size,
        num_inference_steps=args.num_inference_steps,
        noise_type=args.noise_type,
        mask_token_id=mask_token_id,
    )

    pipe = BD3LMPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer)

    print(f"block_size={config.block_size}, mask_token_id={mask_token_id}")
    print(f"Generating {args.gen_length} tokens with {args.num_inference_steps} steps, nucleus_p={args.nucleus_p}")
    print("-" * 50)

    output = pipe(
        gen_length=args.gen_length,
        num_inference_steps=args.num_inference_steps,
        nucleus_p=args.nucleus_p,
        mask_token_id=mask_token_id,
        eos_early_stop=args.eos_early_stop,
        output_type="text",
    )

    print("\nGenerated text:")
    print(output.texts[0] if output.texts else tokenizer.decode(output.sequences[0], skip_special_tokens=True))
    print(f"\nGenerated {output.sequences.shape[1]} tokens")


if __name__ == "__main__":
    main()
