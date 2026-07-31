"""
LTX-2.4 text-to-video parity harness: modular `LTX2Blocks` vs. standard `LTX2Pipeline`.

TEMPORARY / for-visibility only. This whole `integrations/` directory is meant to be committed
transiently while the modular LTX-2 integration is under review, and removed before the final merge.
It is NOT a pytest test and is not wired into CI — run it manually against a real checkpoint.

What it does
------------
Loads `diffusers/LTX-2.4-Diffusers` once, then runs the *same* text-to-video generation through:
  1. the standard `LTX2Pipeline` (`.__call__`), and
  2. the modular `LTX2Blocks` t2v blockset (built with `init_pipeline()`),

reusing the *identical* loaded component objects for both (via `update_components`). Because the
weights are shared, any output difference isolates a difference in the *block logic*, not in loading.

Both runs:
  - use `output_type="latent"` so we compare the denoised latents directly (isolating the joint
    video+audio denoise loop, before VAE decode / vocoder);
  - use a freshly-seeded generator with the same seed (both consume randomness in the same order:
    video noise, then audio noise);
  - disable prompt enhancement (`enable_prompt_enhancement=False` on the standard pipeline; the
    modular `LTX2Blocks` t2v blockset contains no enhancer block), so the raw prompt is used and the
    comparison stays deterministic;
  - apply the *same* guidance settings, delivered differently per pipeline: the standard pipeline takes
    the guidance scales as `__call__` kwargs, while the modular pipeline takes them via its `guider` /
    `audio_guider` components (guidance is guider config in modular diffusers, not a call argument).

Usage
-----
    python integrations/ltx2_t2v_parity.py

Adjust the constants below (checkpoint, resolution, guidance) as needed. Lower `NUM_INFERENCE_STEPS`
keeps the run fast; the fp32 gate holds at any step count (the bf16 gap grows with steps -- see the
DTYPE_TOLERANCES note for why fp32 is the authoritative gate).
"""

import argparse

import torch

from diffusers import LTX2Pipeline
from diffusers.modular_pipelines.ltx2 import LTX2Blocks
from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
from diffusers.pipelines.ltx2 import LTX2AutoDuration
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT


DEFAULT_PROMPT = (
    "A cinematic shot of a red fox walking through a snowy forest at dawn, golden light filtering through pine trees."
)

# Full guidance stack (CFG + spatio-temporal guidance + modality-isolation), matching real LTX-2.4 usage.
# The standard pipeline takes these as `__call__` kwargs; the modular pipeline takes them via its video/audio
# guider components (see `_make_guiders`). Kept identical so the comparison is apples-to-apples.
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
}
# A transformer flag (not guidance), so it stays a `__call__` kwarg for both pipelines.
USE_CROSS_TIMESTEP = True


def _make_guiders(args, guidance: dict) -> tuple[LTX2Guidance, LTX2Guidance]:
    """Build the video/audio guiders from the flat GUIDANCE dict (the modular equivalent of the call kwargs)."""
    video_guider = LTX2Guidance(
        guidance_scale=args.guidance_scale or guidance["guidance_scale"],
        stg_scale=args.stg_scale or guidance["stg_scale"],
        modality_scale=args.mod_scale or guidance["modality_scale"],
        guidance_rescale=args.guidance_rescale or guidance["guidance_rescale"],
        spatio_temporal_guidance_blocks=guidance["spatio_temporal_guidance_blocks"],
    )
    audio_guider = LTX2Guidance(
        guidance_scale=args.audio_guidance_scale or guidance["audio_guidance_scale"],
        stg_scale=args.audio_stg_scale or guidance["audio_stg_scale"],
        modality_scale=args.audio_mod_scale or guidance["audio_modality_scale"],
        guidance_rescale=args.audio_guidance_rescale or guidance["audio_guidance_rescale"],
        # STG blocks are shared (taken from the video guider at plan time); audio only sets its STG *scale*.
    )
    return video_guider, audio_guider


# Components shared between the standard and modular pipelines (everything LTX2Blocks needs for t2v).
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


# Parity tolerances. The modular denoiser runs each guidance pass as its own single-batch transformer forward,
# whereas the standard pipeline batches cond+uncond into one forward. The two are mathematically equivalent, but
# GPU matmul is not batch-invariant -- `cond` computed alone differs from the same `cond` inside a batch-of-2 --
# so the modular and standard latents differ by genuine kernel-reordering noise (amplified by the guidance deltas
# and accumulated over sampler steps), NOT a logic bug. We therefore do NOT gate on assert_close's fp32 defaults:
# rtol=1.3e-6 is effectively a *bitwise* bar, and relative error at near-zero-valued elements explodes without
# meaning (a lone near-zero element can report a huge relative diff). Instead we gate on two magnitude-aware
# statistics, keyed off the *run* dtype:
#   - mean abs diff relative to mean magnitude -- the bulk-agreement signal (a logic bug shifts the whole
#     distribution; kernel noise keeps the mean tiny while a handful of elements drift), and
#   - max abs diff -- a loose ceiling that still catches structured / systematic errors.
# fp32 is the authoritative gate (observed ~1e-4 mean-rel on a full checkpoint). bf16 is a close-but-not-bitwise
# sanity check: its 7-bit mantissa makes the same reordering ~1e4x coarser (~5-10% mean-relative on a full
# checkpoint), so its bounds are loose -- tight enough to catch gross breakage, loose enough to pass the expected
# numerical divergence.
DTYPE_TOLERANCES = {
    # dtype: (mean_rel_tol, max_abs_tol)
    torch.float32: (1e-3, 1e-3),
    torch.bfloat16: (0.15, 0.5),
}


def _tensor_stats(modality: str, std: torch.Tensor, mod: torch.Tensor) -> None:
    print(f"{modality} min std: {std.min()} | mod: {mod.min()}")
    print(f"{modality} mean std: {std.mean()} | mod: {mod.mean()}")
    print(f"{modality} stddev std: {std.std()} | mod: {mod.std()}")
    print(f"{modality} max std: {std.max()} | mod: {mod.max()}")


def _report(name: str, a: torch.Tensor, b: torch.Tensor, mean_rel_tol: float, max_abs_tol: float) -> bool:
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
    mean_rel = mean_abs / denom
    # Gate on bulk agreement (mean abs diff relative to mean magnitude) plus a loose max-abs ceiling, rather than
    # a bitwise assert_close -- see the DTYPE_TOLERANCES note for why the strict fp32 defaults don't fit this
    # single-batch-vs-batched comparison.
    ok_mean = mean_rel <= mean_rel_tol
    ok_max = max_abs <= max_abs_tol
    print(f"  max abs diff:  {max_abs:.3e}   (tol {max_abs_tol:.1e}: {'ok' if ok_max else 'FAIL'})")
    print(
        f"  mean abs diff: {mean_abs:.3e}   mean rel: {mean_rel:.3e}   "
        f"(tol {mean_rel_tol:.1e}: {'ok' if ok_mean else 'FAIL'})"
    )
    ok = ok_mean and ok_max
    print(f"  parity: {'OK' if ok else 'MISMATCH'}")
    return ok


def main(args):
    # 1. Load the standard pipeline once.
    print(f"Loading {args.model_path} ...")
    std = LTX2Pipeline.from_pretrained(args.model_path, torch_dtype=args.dtype)
    if args.cpu_offload:
        std.enable_model_cpu_offload(device=args.device)
    else:
        std.to(args.device)

    if args.predict_duration:
        num_frames = LTX2AutoDuration(min_seconds=args.min_seconds, max_seconds=args.max_seconds)
    else:
        num_frames = args.num_frames
    # Non-guidance call kwargs shared by both pipelines. Guidance is delivered separately: `__call__` kwargs for
    # the standard pipeline, guider components for the modular one.
    common_kwargs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": num_frames,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "num_videos_per_prompt": args.num_videos_per_prompt,
        "use_cross_timestep": USE_CROSS_TIMESTEP,
        "output_type": "latent",
    }

    # 2. Standard run — guidance scales passed as `__call__` kwargs.
    print("Running standard LTX2Pipeline ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    video_std, audio_std = std(
        generator=generator,
        enable_prompt_enhancement=False,  # keep raw prompt for a deterministic comparison
        return_dict=False,
        **common_kwargs,
        **GUIDANCE,
    )

    # 3. Build the modular t2v pipeline reusing the SAME component objects (identical weights), and configure the
    #    guiders with the same guidance settings (guidance is guider config in modular diffusers, not a call kwarg).
    print("Building modular LTX2Blocks pipeline (shared components) ...")
    mod = LTX2Blocks().init_pipeline()
    mod.update_components(**{name: getattr(std, name) for name in SHARED_COMPONENTS})
    if hasattr(std, "duration_head"):
        mod.update_components(duration_head=getattr(std, "duration_head"))
    video_guider, audio_guider = _make_guiders(args, GUIDANCE)
    mod.update_components(guider=video_guider, audio_guider=audio_guider)

    # 4. Modular run — guidance comes from the guiders, so `GUIDANCE` is NOT passed here.
    print("Running modular LTX2Blocks ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    state = mod(generator=generator, **common_kwargs)
    video_mod = state.get("videos")
    audio_mod = state.get("audio")

    # 5. Compare denoised latents.
    if args.check_tensor_stats:
        _tensor_stats("Video", video_std, video_mod)
        _tensor_stats("Audio", audio_std, audio_mod)
    ok_video = _report("video latents", video_std, video_mod, args.mean_rel_tol, args.max_abs_tol)
    ok_audio = _report("audio latents", audio_std, audio_mod, args.mean_rel_tol, args.max_abs_tol)

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
    parser.add_argument("--cpu_offload", action="store_true")

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)

    parser.add_argument("--predict_duration", action="store_true")
    parser.add_argument("--min_seconds", type=float, default=1.0)
    parser.add_argument("--max_seconds", type=float, default=20.0)

    parser.add_argument("--guidance_scale", type=float, default=None, help="Video CFG guidance scale")
    parser.add_argument("--stg_scale", type=float, default=None, help="Video STG guidance scale")
    parser.add_argument("--mod_scale", type=float, default=None, help="Video modality isolation guidance scale")
    parser.add_argument("--guidance_rescale", type=float, default=None, help="Video guidance rescale")
    parser.add_argument("--audio_guidance_scale", type=float, default=None, help="Audio CFG guidance scale")
    parser.add_argument("--audio_stg_scale", type=float, default=None, help="Audio STG guidance scale")
    parser.add_argument("--audio_mod_scale", type=float, default=None, help="Audio modality isolation guidance scale")
    parser.add_argument("--audio_guidance_rescale", type=float, default=None, help="Audio guidance rescale")

    parser.add_argument(
        "--mean_rel_tol",
        type=float,
        default=None,
        help="Parity gate: mean abs diff relative to mean magnitude. Dtype-aware default (fp32: 1e-3, bf16: 0.15).",
    )
    parser.add_argument(
        "--max_abs_tol",
        type=float,
        default=None,
        help="Parity gate: max abs diff ceiling. Dtype-aware default (fp32: 1e-3, bf16: 0.5).",
    )
    parser.add_argument(
        "--check_tensor_stats",
        action="store_true",
        help="Whether to print individual tensor stats for std and modular pipeline outputs",
    )

    args = parser.parse_args()

    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    args.prompt = args.prompt or DEFAULT_PROMPT
    args.negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT

    default_mean_rel_tol, default_max_abs_tol = DTYPE_TOLERANCES[args.dtype]
    if args.mean_rel_tol is None:
        args.mean_rel_tol = default_mean_rel_tol
    if args.max_abs_tol is None:
        args.max_abs_tol = default_max_abs_tol

    main(args)
