#!/usr/bin/env python

import argparse
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from diffusers import HybridTokenDiffusionPipeline, HybridTokenDiffusionScheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Sample with a hybrid-transition token diffusion scheduler.")
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path containing a model + scheduler config."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt; will be used as a fixed prefix.")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--num_inference_steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--inject_start_token", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.checkpoint_path).to(device)
    scheduler = HybridTokenDiffusionScheduler.from_pretrained(args.checkpoint_path)

    pipe = HybridTokenDiffusionPipeline(model=model, scheduler=scheduler, tokenizer=tokenizer).to(device)
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
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        prefix_ids=prefix_ids,
        inject_start_token=args.inject_start_token,
        return_text=True,
    )

    for i, t in enumerate(out.texts or []):
        print(f"[{i}] {t}")


if __name__ == "__main__":
    main()
