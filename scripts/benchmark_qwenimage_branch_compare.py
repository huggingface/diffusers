import argparse
import json
import os
import subprocess
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark QwenImage pipeline runtime.")
    parser.add_argument("--repo-root", default=None, help="Path to diffusers repo to benchmark.")
    parser.add_argument("--model-path", required=True, help="Local path or hub id for QwenImage.")
    parser.add_argument("--prompt", default="A serene mountain landscape at sunset", help="Prompt text.")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-backend", default=None, help="Set DIFFUSERS_ATTN_BACKEND if provided.")
    parser.add_argument("--output-json", required=True, help="Output JSON path.")
    return parser.parse_args()


def _git_rev(repo_root: str | None) -> str:
    if not repo_root:
        return ""
    try:
        out = subprocess.check_output(["git", "-C", repo_root, "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return ""


def main() -> None:
    args = parse_args()

    if args.repo_root:
        sys.path.insert(0, args.repo_root)

    if args.attn_backend:
        os.environ["DIFFUSERS_ATTN_BACKEND"] = args.attn_backend

    from diffusers import QwenImagePipeline

    device = torch.device(args.device)

    pipe = QwenImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    inputs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
    }

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = pipe(**inputs)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = pipe(**inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    result = {
        "repo_root": args.repo_root,
        "git_rev": _git_rev(args.repo_root),
        "model_path": args.model_path,
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "attn_backend": args.attn_backend,
        "times": times,
        "mean_seconds": mean,
        "std_seconds": std,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved results to {args.output_json}")
    print(f"Mean: {mean:.4f}s, Std: {std:.4f}s")


if __name__ == "__main__":
    main()
