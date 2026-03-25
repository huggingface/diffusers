#!/usr/bin/env python
"""Re-save RAE checkpoints with updated weights (includes latent normalization buffers)
and push as PRs to the HuggingFace Hub.

Usage:
    python scripts/push_rae_updated_weights.py
    python scripts/push_rae_updated_weights.py --dry-run  # load and save locally only
"""

import argparse
import tempfile

import torch

from diffusers import AutoencoderRAE


REPOS = [
    "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08",
    "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08-i512",
    "nyu-visionx/RAE-dinov2-wReg-small-ViTXL-n08",
    "nyu-visionx/RAE-dinov2-wReg-large-ViTXL-n08",
    "nyu-visionx/RAE-siglip2-base-p16-i256-ViTXL-n08",
    "nyu-visionx/RAE-mae-base-p16-ViTXL-n08",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only load and save locally, do not push.")
    parser.add_argument("--repos", nargs="+", default=REPOS, help="Repos to update (default: all 6).")
    args = parser.parse_args()

    for repo_id in args.repos:
        print(f"\n{'=' * 60}")
        print(f"Processing: {repo_id}")
        print(f"{'=' * 60}")

        model = AutoencoderRAE.from_pretrained(repo_id)
        model.eval()

        # Quick sanity check
        enc_input_size = model.config.get("encoder_input_size", 224) or 224
        x = torch.rand(1, 3, enc_input_size, enc_input_size)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out.sample).all(), f"Non-finite output for {repo_id}"
        print(f"  Sanity check passed: output shape {out.sample.shape}")

        if args.dry_run:
            with tempfile.TemporaryDirectory() as tmpdir:
                model.save_pretrained(tmpdir)
                print(f"  Dry run: saved to {tmpdir}")
        else:
            model.push_to_hub(
                repo_id,
                commit_message="Update weights: include latent normalization buffers for diffusers compatibility",
                create_pr=True,
            )
            print(f"  PR created for {repo_id}")


if __name__ == "__main__":
    main()
