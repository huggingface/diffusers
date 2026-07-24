"""
LTX-2.4 image-to-video parity harness: modular `LTX2ImageToVideoBlocks` vs. standard
`LTX2ImageToVideoPipeline`.

TEMPORARY / for-visibility only. This whole `integrations/` directory is meant to be committed
transiently while the modular LTX-2 integration is under review, and removed before the final merge.
It is NOT a pytest test and is not wired into CI — run it manually against a real checkpoint.

What it does
------------
Loads `diffusers/LTX-2.4-Diffusers` once, then runs the *same* image-to-video generation through:
  1. the standard `LTX2ImageToVideoPipeline` (`.__call__`), and
  2. the modular `LTX2ImageToVideoBlocks` i2v blockset (built with `init_pipeline()`),

reusing the *identical* loaded component objects for both (via `update_components`). Because the
weights are shared, any output difference isolates a difference in the *block logic*, not in loading.

Both runs:
  - use the same reference `image` (a deterministic synthetic gradient by default — swap in a real
    image via `load_image` if you prefer);
  - use `output_type="latent"` so we compare the denoised latents directly (isolating the joint
    video+audio denoise loop with image conditioning, before VAE decode / vocoder);
  - use a freshly-seeded generator with the same seed. The image is VAE-encoded with `argmax` (mode,
    no randomness), so both pipelines then consume randomness in the same order: video noise, audio
    noise;
  - disable prompt enhancement (`enable_prompt_enhancement=False` on the standard pipeline; the
    modular `LTX2ImageToVideoBlocks` i2v blockset contains no enhancer block).

Usage
-----
    python integrations/ltx2_i2v_parity.py

Adjust the constants below (checkpoint, resolution, guidance, image) as needed. Lower
`NUM_INFERENCE_STEPS` keeps the run fast; parity holds at any step count.
"""

import argparse

import numpy as np
import torch
from PIL import Image

from diffusers import LTX2ImageToVideoPipeline
from diffusers.modular_pipelines.ltx2 import LTX2ImageToVideoBlocks
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image


DEFAULT_PROMPT = "The fox turns its head and blinks slowly as snow begins to fall."

# Full guidance stack (CFG + spatio-temporal guidance + modality-isolation), matching real LTX-2.4 usage.
GUIDANCE = {
    "guidance_scale": 3.0,
    "stg_scale": 1.0,
    "modality_scale": 3.0,
    "guidance_rescale": 0.7,
    "audio_guidance_scale": 7.0,
    "audio_stg_scale": 1.0,
    "audio_modality_scale": 3.0,
    "audio_guidance_rescale": 0.7,
    "spatio_temporal_guidance_blocks": [29],
    "use_cross_timestep": True,
}

# Components shared between the standard and modular pipelines (everything LTX2ImageToVideoBlocks needs).
SHARED_COMPONENTS = [
    "text_encoder",
    "tokenizer",
    "connectors",
    "transformer",
    "vae",
    "audio_vae",
    "vocoder",
    "scheduler",
    "video_processor",
]


def _reference_image(width: int, height: int) -> Image.Image:
    # Deterministic synthetic gradient — no network dependency. The content is irrelevant to parity
    # (both pipelines receive the exact same image). To use a real image instead:
    #   from diffusers.utils import load_image
    #   return load_image("<url-or-path>")
    yy, xx = np.mgrid[0:height, 0:width]
    r = (xx / width * 255).astype(np.uint8)
    g = (yy / height * 255).astype(np.uint8)
    b = ((xx + yy) / (width + height) * 255).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1))


# Per-dtype default tolerances, matching torch.testing.assert_close's built-in defaults
# (fp32: rtol=1.3e-6, atol=1e-5; bf16: rtol=1.6e-2, atol=1e-5). Keyed off the *run* dtype rather
# than the upcast comparison dtype, so a bf16 run isn't spuriously held to fp32 strictness.
DTYPE_TOLERANCES = {
    torch.float32: (1.3e-6, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
}


def _report(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> bool:
    a = a.float().cpu()
    b = b.float().cpu()
    print(f"\n[{name}]")
    print(f"  standard shape: {tuple(a.shape)}   modular shape: {tuple(b.shape)}")
    if a.shape != b.shape:
        print("  SHAPE MISMATCH")
        return False
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    denom = a.abs().mean().item() or 1.0
    print(f"  max abs diff:  {max_abs:.3e}")
    print(f"  mean abs diff: {mean_abs:.3e}   (relative to mean magnitude: {mean_abs / denom:.3e})")
    # Compare with explicit (dtype-aware) tolerances rather than assert_close's own dtype inference,
    # since the tensors were upcast to float above. assert_close's failure message reports the count
    # of mismatched elements and the greatest absolute/relative diff, which is more useful than a bool.
    try:
        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        ok = True
    except AssertionError as err:
        ok = False
        for line in str(err).strip().splitlines():
            print(f"  {line}")
    print(f"  assert_close(atol={atol:.1e}, rtol={rtol:.1e}): {ok}")
    return ok


def main(args):
    if args.image is not None:
        image = load_image(args.image)
    else:
        image = _reference_image(args.width, args.height)

    # 1. Load the standard pipeline once.
    print(f"Loading {args.model_path} ...")
    std = LTX2ImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=args.dtype).to(args.device)

    common_kwargs = {
        "image": image,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "output_type": "latent",
        **GUIDANCE,
    }

    # 2. Standard run.
    print("Running standard LTX2ImageToVideoPipeline ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    video_std, audio_std = std(
        generator=generator,
        enable_prompt_enhancement=False,  # keep raw prompt for a deterministic comparison
        return_dict=False,
        **common_kwargs,
    )

    # 3. Build the modular i2v pipeline reusing the SAME component objects (identical weights).
    print("Building modular LTX2ImageToVideoBlocks pipeline (shared components) ...")
    mod = LTX2ImageToVideoBlocks().init_pipeline()
    mod.update_components(**{name: getattr(std, name) for name in SHARED_COMPONENTS})

    # 4. Modular run.
    print("Running modular LTX2ImageToVideoBlocks ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    state = mod(generator=generator, **common_kwargs)
    video_mod = state.get("videos")
    audio_mod = state.get("audio")

    # 5. Compare denoised latents.
    ok_video = _report("video latents", video_std, video_mod, atol=args.atol, rtol=args.rtol)
    ok_audio = _report("audio latents", audio_std, audio_mod, atol=args.atol, rtol=args.rtol)

    print("\n" + ("PARITY OK" if (ok_video and ok_audio) else "PARITY MISMATCH — investigate above"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path loadable by from_pretrained",
    )

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--image", type=str, default=None)

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--max_sequence_length", type=int, default=1024)

    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Absolute tolerance for assert_close; defaults to a dtype-aware value (fp32/bf16: 1e-5).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Relative tolerance for assert_close; defaults to a dtype-aware value (fp32: 1.3e-6, bf16: 1.6e-2).",
    )

    args = parser.parse_args()

    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    args.prompt = args.prompt or DEFAULT_PROMPT
    args.negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    default_rtol, default_atol = DTYPE_TOLERANCES[args.dtype]
    if args.atol is None:
        args.atol = default_atol
    if args.rtol is None:
        args.rtol = default_rtol

    main(args)
