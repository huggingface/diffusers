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
Sample script for LLaDA2-style discrete diffusion text generation.

This script demonstrates how to use the LLaDA2Pipeline for text generation
using block-wise iterative refinement.

Example usage:
    python sample_llada2.py --model_id inclusionAI/LLaDA2.0-mini --prompt "What is the capital of France?"
    python sample_llada2.py --model_id inclusionAI/LLaDA2.0-flash-CAP --prompt "Explain quantum computing." --temperature 0.7
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import LLaDA2Pipeline
from diffusers.hooks import apply_group_offloading


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using LLaDA2Pipeline with block-wise discrete diffusion."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="inclusionAI/LLaDA2.0-mini",
        help="HuggingFace model ID or path to local model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Why does Camus think that Sisyphus is happy?",
        help="Text prompt to generate from.",
    )
    parser.add_argument(
        "--gen_length",
        type=int,
        default=2048,
        help="Number of tokens to generate.",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Size of each generation block.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=32,
        help="Number of refinement steps per block.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy).",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Nucleus sampling probability threshold.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Confidence threshold for committing tokens.",
    )
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="multinomial",
        choices=["auto", "greedy", "multinomial"],
        help="Sampling method for block refinement.",
    )
    parser.add_argument(
        "--eos_early_stop",
        action="store_true",
        help="Stop generation early when EOS token is generated.",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--offload",
        type=str,
        default=None,
        choices=["group", "sequential"],
        help="Memory offloading strategy: 'group' for group offloading (faster), 'sequential' for sequential CPU offload (slower but lower memory).",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)

    # Load model with appropriate memory settings based on offload strategy
    if args.offload == "group":
        # For group offloading, load to CPU first then apply hooks
        print("Using group offloading for memory efficiency...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        # Apply group offloading with CUDA streams for better performance
        onload_device = torch.device(args.device)
        offload_device = torch.device("cpu")
        apply_group_offloading(
            model,
            onload_device=onload_device,
            offload_device=offload_device,
            offload_type="leaf_level",
            use_stream=True,
        )
    elif args.offload == "sequential":
        # For sequential offloading, load to CPU first
        print("Using sequential CPU offloading (slower but lower memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        # Sequential offloading will be applied via pipeline
    else:
        # Default: use device_map="auto" for automatic memory management
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
    model.eval()

    # Create pipeline
    pipe = LLaDA2Pipeline(model=model, tokenizer=tokenizer)

    # Apply sequential CPU offload if requested
    if args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()

    # Set up generator for reproducibility
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.gen_length} tokens with block_length={args.block_length}, steps={args.steps}")
    print("-" * 50)

    # Generate
    output = pipe(
        prompt=args.prompt,
        use_chat_template=args.use_chat_template,
        add_generation_prompt=args.add_generation_prompt,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        threshold=args.threshold,
        sampling_method=args.sampling_method,
        eos_early_stop=args.eos_early_stop,
        generator=generator,
    )

    print("\nGenerated text:")
    print(output.texts[0])

    print(f"\nGenerated {output.sequences.shape[1]} tokens")


if __name__ == "__main__":
    main()
