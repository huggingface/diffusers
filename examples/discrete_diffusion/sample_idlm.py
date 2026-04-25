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
Sample script for I-DLM (Introspective Diffusion Language Model) block-N decoding.

Example:
    python sample_idlm.py \
      --model_id yifanyu/I-DLM-8B \
      --prompt "Prove that sqrt(2) is irrational." \
      --gen_block_size 4 \
      --max_new_tokens 256
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import IDLMBlockDiffusionScheduler, IDLMPipeline


def main():
    parser = argparse.ArgumentParser(description="Run I-DLM introspective strided decoding.")
    parser.add_argument("--model_id", type=str, default="yifanyu/I-DLM-8B", help="Model ID or local path.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Prove that sqrt(2) is irrational.",
        help="Prompt text to generate from.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--gen_block_size",
        type=int,
        default=4,
        help="Block size N: each ISD round commits up to N tokens. `block_size = 2*N - 1`.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--verify_alpha",
        type=float,
        default=1.0,
        help="Leniency in the min(1, p/(alpha*q)) accept criterion. 1.0 = standard verify.",
    )
    parser.add_argument("--mask_token_id", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--add_generation_prompt", action="store_true")
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable <think>...</think> reasoning in the chat template. I-DLM is a Qwen3-derivative; "
        "thinking is ON by default in the Qwen3 template. Pass this flag to keep it on.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype)

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        dtype=torch_dtype if torch_dtype is not None else "auto",
        device_map=args.device,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, revision=args.revision)
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    scheduler = IDLMBlockDiffusionScheduler(
        gen_block_size=args.gen_block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verify_alpha=args.verify_alpha,
    )
    pipe = IDLMPipeline(model=model, tokenizer=tokenizer, scheduler=scheduler)

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"\nPrompt: {args.prompt}")
    chat_template_kwargs = {"enable_thinking": bool(args.enable_thinking)}
    output = pipe(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        gen_block_size=args.gen_block_size,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        verify_alpha=args.verify_alpha,
        mask_token_id=args.mask_token_id,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=args.add_generation_prompt,
        chat_template_kwargs=chat_template_kwargs,
        generator=generator,
    )

    print("\nGenerated text:")
    print(
        output.texts[0]
        if output.texts is not None
        else tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    )
    print(f"\nGenerated {output.sequences.shape[1]} tokens")


if __name__ == "__main__":
    main()
