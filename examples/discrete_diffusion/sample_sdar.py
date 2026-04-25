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
Sample script for SDAR-style block diffusion decoding.

Example:
    python sample_sdar.py \
      --model_id JetLM/SDAR-1.7B-Chat \
      --prompt "Explain what reinforcement learning is in simple terms." \
      --max_new_tokens 256
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import SDARPipeline


def main():
    parser = argparse.ArgumentParser(description="Run SDAR block diffusion decoding.")
    parser.add_argument(
        "--model_id",
        type=str,
        default="JetLM/SDAR-1.7B-Chat",
        help="Model ID or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain what reinforcement learning is in simple terms.",
        help="Prompt text to generate from.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=4)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--remasking_strategy",
        type=str,
        default="low_confidence_dynamic",
        choices=["low_confidence_dynamic", "low_confidence_static", "sequential", "entropy_bounded"],
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--entropy_threshold", type=float, default=0.35)
    parser.add_argument("--mask_token_id", type=int, default=None)
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use the tokenizer chat template for the prompt.",
    )
    parser.add_argument(
        "--add_generation_prompt",
        action="store_true",
        help="Add the generation prompt when using the chat template.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Model revision (branch, tag, or commit hash).",
    )

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype)

    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype if torch_dtype is not None else "auto",
        device_map=args.device,
        revision=args.revision,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, revision=args.revision)

    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})

    pipe = SDARPipeline(model=model, tokenizer=tokenizer)

    print(f"\nPrompt: {args.prompt}")
    output = pipe(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        block_length=args.block_length,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        remasking_strategy=args.remasking_strategy,
        confidence_threshold=args.confidence_threshold,
        entropy_threshold=args.entropy_threshold,
        mask_token_id=args.mask_token_id,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=args.add_generation_prompt,
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
