"""Merge a LoRA adapter into a Diffusers pipeline using its supported adapter loader."""

import argparse
from pathlib import Path

from diffusers import DiffusionPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base Diffusers pipeline directory or Hub ID")
    parser.add_argument("--adapter", required=True, help="LoRA file, directory or Hub ID")
    parser.add_argument("--output", required=True, help="New output pipeline directory")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    pipeline = DiffusionPipeline.from_pretrained(args.model)
    pipeline.load_lora_weights(args.adapter)
    pipeline.fuse_lora(lora_scale=args.scale, safe_fusing=True)
    pipeline.unload_lora_weights()
    pipeline.save_pretrained(args.output, safe_serialization=True)


if __name__ == "__main__":
    main()
