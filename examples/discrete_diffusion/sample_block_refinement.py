#!/usr/bin/env python

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import BlockRefinementPipeline


def main():
    parser = argparse.ArgumentParser(description="Sample with BlockRefinementPipeline using a transformers causal LM.")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Write a short paragraph about diffusion models.")
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention_mask_mode", type=str, default="2d", choices=["auto", "4d", "2d", "none"])

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        cache_dir=args.cache_dir,
    )
    model.to(args.device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have `mask_token_id` for block refinement sampling.")

    pipe = BlockRefinementPipeline(model=model, tokenizer=tokenizer).to(args.device)
    gen = torch.Generator(device=args.device).manual_seed(args.seed)

    prompt_ids = tokenizer(args.prompt, return_tensors="pt")["input_ids"].to(args.device)
    out = pipe(
        prompt_ids=prompt_ids,
        gen_length=int(args.gen_length),
        block_length=int(args.block_length),
        steps=int(args.steps),
        temperature=float(args.temperature),
        top_p=None if args.top_p >= 1.0 else float(args.top_p),
        top_k=None if args.top_k <= 0 else int(args.top_k),
        threshold=float(args.threshold),
        eos_early_stop=True,
        eos_token_id=int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        mask_token_id=int(tokenizer.mask_token_id),
        attention_mask_mode=args.attention_mask_mode,
        generator=gen,
        return_text=True,
    )

    print(out.texts[0] if out.texts is not None else tokenizer.decode(out.sequences[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
