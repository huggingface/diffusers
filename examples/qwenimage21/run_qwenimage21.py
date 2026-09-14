"""Sample script for Qwen-Image 2.1: text-to-image and image editing.

Both tasks run 40 denoising steps with classifier-free guidance off, so each step is a single
forward pass through the transformer, and with the prefix KV cache on.

    # both tasks; `edit` reuses the text-to-image result as its condition image
    python run_qwenimage21.py --model Qwen/Qwen-Image-2.1

    # editing your own image
    python run_qwenimage21.py --task edit --image cat.png --edit-prompt "make it snow"

Requires `diffusers` with Qwen-Image 2.1 support, plus `transformers`, `accelerate` and `torch`.
Block-causal attention uses `torch.nn.attention.flex_attention` when the installed torch provides
it; otherwise the script still runs on the two-pass SDPA fallback, which is exact for the target
image but only approximate for condition images.
"""

import argparse
import time
from pathlib import Path

import torch

from diffusers import QwenImage21Pipeline
from diffusers.utils import load_image


T2I_PROMPT = "A capybara wearing a wizard hat, reading a book by candlelight, oil painting"
EDIT_PROMPT = "Move the capybara to a snowy mountain top at sunrise, keep the wizard hat"

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Classifier-free guidance is off: `true_cfg_scale <= 1` skips the negative-prompt pass entirely.
TRUE_CFG_SCALE = 1.0

# The text and condition-image prefix is modulated from `t = 0`, so its keys and values are step-independent and are
# cached after the first step. Required for these samples, so it is not exposed as a flag.
USE_KV_CACHE = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen-Image-2.1", help="Hub repo id or local checkpoint directory.")
    parser.add_argument(
        "--task", choices=["t2i", "edit", "both"], default="both", help="Which sample(s) to run. Default: both."
    )
    parser.add_argument("--prompt", default=T2I_PROMPT, help="Text-to-image prompt.")
    parser.add_argument("--edit-prompt", default=EDIT_PROMPT, help="Instruction applied to the condition image.")
    parser.add_argument(
        "--image",
        default=None,
        help="Condition image for `edit` (path or URL). Defaults to the text-to-image result, "
        "which is generated first if needed.",
    )
    parser.add_argument("--steps", type=int, default=40, help="Denoising steps. Default: 40.")
    parser.add_argument("--resolution", type=int, default=2048, help="Target side length in pixels. Default: 1024.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the latent noise.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bf16")
    parser.add_argument("--output-dir", type=Path, default=Path("qwenimage21_samples"))
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep components on CPU until needed. Cuts peak VRAM at the cost of speed.",
    )
    return parser.parse_args()


def flex_attention_available():
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
    except ImportError:
        return False
    return True


def load_pipeline(args):
    print(f"Loading {args.model} ({args.dtype}) ...")
    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype])
    # The pipeline silently drops the cache when the checkpoint sets `causal_condition=False`, since the prefix is then
    # modulated from the sampled timestep and its keys and values change every step. Fail instead of running uncached.
    if not pipe.transformer.config.causal_condition:
        raise ValueError(
            f"{args.model} was configured with `causal_condition=False`, which makes the prefix KV cache invalid. "
            "This script requires the cache."
        )
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(device=args.device)
    else:
        pipe.to(args.device)
    return pipe


def run(pipe, args, tag, prompt, image=None):
    label = "edit" if image is not None else "text-to-image"
    print(f"\n[{label}] {args.steps} steps, true_cfg_scale={TRUE_CFG_SCALE} (CFG off), kv cache on")
    print(f"[{label}] prompt: {prompt}")

    # A CPU generator keeps the sample reproducible for a given seed on any device.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=args.steps,
        true_cfg_scale=TRUE_CFG_SCALE,
        output_resolution=args.resolution,
        use_kv_cache=USE_KV_CACHE,
        generator=generator,
    )
    elapsed = time.perf_counter() - start

    out_path = args.output_dir / f"{tag}.png"
    result.images[0].save(out_path)
    report = f"[{label}] {elapsed:.1f}s ({elapsed / args.steps:.2f}s/step) -> {out_path}"
    if args.device.startswith("cuda"):
        report += f", peak VRAM {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB"
    print(report)
    return result.images[0]


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"torch {torch.__version__}, device {args.device}")
    print(f"flex_attention: {'available' if flex_attention_available() else 'unavailable, using SDPA fallback'}")

    pipe = load_pipeline(args)

    t2i_image = None
    if args.task in ("t2i", "both"):
        t2i_image = run(pipe, args, "t2i", args.prompt)

    if args.task in ("edit", "both"):
        if args.image is not None:
            condition = load_image(args.image)
        elif t2i_image is not None:
            condition = t2i_image
        else:
            # `--task edit` on its own with no `--image`: produce a condition image first.
            condition = run(pipe, args, "t2i", args.prompt)
        run(pipe, args, "edit", args.edit_prompt, image=condition)

    print(f"\nDone. Images written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
