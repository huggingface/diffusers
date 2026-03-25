import argparse
import json
import subprocess
import sys
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark QwenImageEditPlus pipeline runtime.")
    parser.add_argument("--repo-root", default=None, help="Path to diffusers repo to benchmark.")
    parser.add_argument("--model-id", default="Qwen/Qwen-Image-Edit-2511", help="Model id or local path.")
    parser.add_argument(
        "--prompt",
        default=(
            "The magician bear is on the left, the alchemist bear is on the right, "
            "facing each other in the central park square."
        ),
        help="Prompt text.",
    )
    parser.add_argument(
        "--image-1",
        default="https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_1.jpg",
        help="First input image path or URL.",
    )
    parser.add_argument(
        "--image-2",
        default="https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg",
        help="Second input image path or URL.",
    )
    parser.add_argument("--height", type=int, default=None, help="Optional output height.")
    parser.add_argument("--width", type=int, default=None, help="Optional output width.")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
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

    from diffusers import QwenImageEditPlusPipeline
    from diffusers.utils import load_image

    device = torch.device(args.device)

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    print(f"Running on device={device}, dtype={pipe.dtype}")

    image1 = load_image(args.image_1)
    image2 = load_image(args.image_2)

    generator = torch.Generator(device=device).manual_seed(0)

    inputs = {
        "image": [image1, image2],
        "prompt": args.prompt,
        "generator": generator,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_inference_steps": args.steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    if args.height is not None:
        inputs["height"] = args.height
    if args.width is not None:
        inputs["width"] = args.width

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
        "model_id": args.model_id,
        "prompt": args.prompt,
        "image_1": args.image_1,
        "image_2": args.image_2,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "device": str(device),
        "dtype": str(pipe.dtype),
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
