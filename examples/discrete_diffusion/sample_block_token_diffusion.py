#!/usr/bin/env python

import argparse
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from diffusers import BlockTokenDiffusionPipeline, BlockTokenDiffusionScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample with block-wise token diffusion.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path saved by train scripts (or compatible)."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt; will be used as a fixed prefix.")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--num_inference_steps", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inject_start_token", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path).to(device)
    scheduler = BlockTokenDiffusionScheduler.from_pretrained(args.checkpoint_path)

    pipe = BlockTokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer).to(device)
    model.eval()

    generator: Optional[torch.Generator] = torch.Generator(device=device).manual_seed(args.seed)

    prefix_ids = None
    if args.prompt is not None:
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
        prefix_ids = encoded["input_ids"].to(device=device, dtype=torch.long)
        if prefix_ids.shape[1] > args.seq_len:
            raise ValueError(f"--seq_len ({args.seq_len}) must be >= prompt length ({prefix_ids.shape[1]}).")

    out = pipe(
        batch_size=args.num_samples,
        seq_len=args.seq_len,
        block_size=args.block_size,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        prefix_ids=prefix_ids,
        inject_start_token=args.inject_start_token,
        top_p=args.top_p,
        return_text=True,
    )

    for i, t in enumerate(out.texts or []):
        print(f"[{i}] {t}")


if __name__ == "__main__":
    main()
