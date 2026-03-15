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
Sample script for DFlash speculative decoding.

Example:
    python sample_dflash.py \
      --draft_model_id z-lab/Qwen3-8B-DFlash-b16 \
      --target_model_id Qwen/Qwen3-8B \
      --prompt "How many positive whole-number divisors does 196 have?" \
      --max_new_tokens 256
"""

import argparse

import torch

from diffusers import DFlashPipeline


def main():
    parser = argparse.ArgumentParser(description="Run DFlash speculative decoding.")
    parser.add_argument(
        "--draft_model_id",
        type=str,
        default="z-lab/Qwen3-8B-DFlash-b16",
        help="Draft model ID or local path.",
    )
    parser.add_argument(
        "--target_model_id",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Target model ID or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How many positive whole-number divisors does 196 have?",
        help="Prompt text to generate from.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
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
        "--enable_thinking",
        action="store_true",
        help="Enable chat-template thinking mode if supported by the tokenizer.",
    )
    parser.add_argument(
        "--mask_token",
        type=str,
        default="<|MASK|>",
        help="Mask token to add if the tokenizer does not define one.",
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

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(args.dtype)

    print(f"Loading draft model: {args.draft_model_id}")
    print(f"Loading target model: {args.target_model_id}")
    dtype_arg = torch_dtype if torch_dtype is not None else "auto"
    pipe = DFlashPipeline.from_pretrained(
        draft_model_id=args.draft_model_id,
        target_model_id=args.target_model_id,
        mask_token=args.mask_token,
        draft_model_kwargs={
            "trust_remote_code": True,
            "dtype": dtype_arg,
            "device_map": args.device,
        },
        target_model_kwargs={
            "dtype": dtype_arg,
            "device_map": args.device,
        },
    )

    chat_kwargs = {"enable_thinking": args.enable_thinking}

    print(f"\nPrompt: {args.prompt}")
    output = pipe(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=args.add_generation_prompt,
        chat_template_kwargs=chat_kwargs,
    )

    print("\nGenerated text:")
    print(output.texts[0])
    print(f"\nGenerated {output.sequences.shape[1]} tokens")


if __name__ == "__main__":
    main()
