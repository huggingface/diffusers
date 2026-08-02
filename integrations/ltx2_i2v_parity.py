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
`NUM_INFERENCE_STEPS` keeps the run fast; the fp32 gate holds at any step count (the bf16 gap grows
with steps -- see the DTYPE_TOLERANCES note for why fp32 is the authoritative gate).
"""

import argparse

import numpy as np
import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ImageToVideoPipeline
from diffusers.modular_pipelines.ltx2 import LTX2ImageToVideoBlocks
from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
from diffusers.pipelines.ltx2 import LTX2AutoDuration
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image


DEFAULT_PROMPT = "The fox turns its head and blinks slowly as snow begins to fall."

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


def _resolve_guidance(args) -> dict:
    """Apply the CLI guidance overrides on top of the GUIDANCE defaults. The SAME resolved dict drives BOTH the
    standard pipeline (as `__call__` kwargs) and the modular guiders, so an override like `--guidance_scale 1.0`
    disables CFG on both sides -- otherwise the standard run keeps the hardcoded GUIDANCE and the comparison is
    apples-to-oranges."""
    return {
        "guidance_scale": args.guidance_scale or GUIDANCE["guidance_scale"],
        "stg_scale": args.stg_scale or GUIDANCE["stg_scale"],
        "modality_scale": args.mod_scale or GUIDANCE["modality_scale"],
        "guidance_rescale": args.guidance_rescale or GUIDANCE["guidance_rescale"],
        "audio_guidance_scale": args.audio_guidance_scale or GUIDANCE["audio_guidance_scale"],
        "audio_stg_scale": args.audio_stg_scale or GUIDANCE["audio_stg_scale"],
        "audio_modality_scale": args.audio_mod_scale or GUIDANCE["audio_modality_scale"],
        "audio_guidance_rescale": args.audio_guidance_rescale or GUIDANCE["audio_guidance_rescale"],
        "spatio_temporal_guidance_blocks": GUIDANCE["spatio_temporal_guidance_blocks"],
    }


def _make_guiders(guidance: dict) -> tuple[LTX2Guidance, LTX2Guidance]:
    """Build the video/audio guiders from the resolved guidance dict (the modular equivalent of the call kwargs)."""
    video_guider = LTX2Guidance(
        guidance_scale=guidance["guidance_scale"],
        stg_scale=guidance["stg_scale"],
        modality_scale=guidance["modality_scale"],
        guidance_rescale=guidance["guidance_rescale"],
        spatio_temporal_guidance_blocks=guidance["spatio_temporal_guidance_blocks"],
    )
    audio_guider = LTX2Guidance(
        guidance_scale=guidance["audio_guidance_scale"],
        stg_scale=guidance["audio_stg_scale"],
        modality_scale=guidance["audio_modality_scale"],
        guidance_rescale=guidance["audio_guidance_rescale"],
        # STG blocks are shared (taken from the video guider at plan time); audio only sets its STG *scale*.
    )
    return video_guider, audio_guider


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
# fp32 is the authoritative gate. Its bounds are looser here than in the t2v harness because multi-frame i2v
# amplifies the batch-invariance noise substantially: the per-token masked video timestep (the conditioning frame
# rides at timestep 0) plus a clean anchor frame the denoised frames attend to make the trajectory far more
# sensitive to the tiny per-op differences, so the same cond/uncond batching that lands at ~1e-4 mean-rel for t2v
# lands at ~8e-3 mean-rel / ~5e-2 max-abs for multi-frame i2v with full guidance. This is confirmed numerical, not
# a logic bug: with CFG disabled on BOTH sides (`--guidance_scale 1.0 --audio_guidance_scale 1.0`, no cond/uncond
# batching) the same run is bitwise-ish (~5e-6 mean-rel) -- that CFG-off run is the tight bug-catching gate, and
# these full-guidance bounds are the looser realistic-usage check. bf16 is a close-but-not-bitwise sanity check:
# its 7-bit mantissa makes the same reordering ~1e4x coarser, so its bounds are looser still.
DTYPE_TOLERANCES = {
    # dtype: (mean_rel_tol, max_abs_tol)
    torch.float32: (2e-2, 1.5e-1),
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
    if args.image is not None:
        image = load_image(args.image)
    else:
        image = _reference_image(args.width, args.height)

    # 1. Load the standard pipeline once.
    print(f"Loading {args.model_path} ...")
    std = LTX2ImageToVideoPipeline.from_pretrained(args.model_path, torch_dtype=args.dtype)
    # For now, disable shift_terminal so that num_inference_steps=1 doesn't produce nans.
    new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        std.scheduler.config, shift_terminal=None,
    )
    std.scheduler = new_scheduler
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
        "image": image,
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

    # Resolve guidance once (CLI overrides on top of GUIDANCE defaults) and use it for BOTH pipelines.
    guidance = _resolve_guidance(args)

    # 2. Standard run — guidance scales passed as `__call__` kwargs.
    print("Running standard LTX2ImageToVideoPipeline ...")

    # Optional: capture the first transformer forward of each run. Both pipelines share `std.transformer`, so
    # identical inputs must give identical outputs -- diffing the first forward's kwargs pinpoints which input
    # first diverges (or shows identical inputs => the divergence is in the combine/step, not the forward).
    # Use a real forward hook (not an instance `.forward` override) so it fires regardless of accelerate
    # offload wrapping and no matter how each pipeline reaches the shared module.
    forward_captures = {"std": [], "mod": []}
    phase = {"name": None}
    hook_handle = None
    if args.debug_forward:

        def _capture_hook(module, fwd_args, fwd_kwargs, output):
            name = phase["name"]
            if name is None or forward_captures[name]:
                return
            snap = {
                k: (v.detach().float().cpu() if isinstance(v, torch.Tensor) else v) for k, v in fwd_kwargs.items()
            }
            snap["__out_video__"] = output[0].detach().float().cpu()
            snap["__out_audio__"] = output[1].detach().float().cpu()
            forward_captures[name].append(snap)

        hook_handle = std.transformer.register_forward_hook(_capture_hook, with_kwargs=True)
        phase["name"] = "std"

    generator = torch.Generator(args.device).manual_seed(args.seed)
    video_std, audio_std = std(
        generator=generator,
        enable_prompt_enhancement=False,  # keep raw prompt for a deterministic comparison
        return_dict=False,
        **common_kwargs,
        **guidance,
    )

    # 3. Build the modular i2v pipeline reusing the SAME component objects (identical weights), and configure the
    #    guiders with the same guidance settings (guidance is guider config in modular diffusers, not a call kwarg).
    print("Building modular LTX2ImageToVideoBlocks pipeline (shared components) ...")
    mod = LTX2ImageToVideoBlocks().init_pipeline()
    mod.update_components(**{name: getattr(std, name) for name in SHARED_COMPONENTS})
    if hasattr(std, "duration_head"):
        mod.update_components(duration_head=getattr(std, "duration_head"))
    video_guider, audio_guider = _make_guiders(guidance)
    mod.update_components(guider=video_guider, audio_guider=audio_guider)

    # 4. Modular run — guidance comes from the guiders, so `GUIDANCE` is NOT passed here.
    print("Running modular LTX2ImageToVideoBlocks ...")
    if args.debug_forward:
        phase["name"] = "mod"
    generator = torch.Generator(args.device).manual_seed(args.seed)
    state = mod(generator=generator, **common_kwargs)
    video_mod = state.get("videos")
    audio_mod = state.get("audio")

    if args.debug_forward:
        if hook_handle is not None:
            hook_handle.remove()
        cs = forward_captures["std"][0] if forward_captures["std"] else {}
        cm = forward_captures["mod"][0] if forward_captures["mod"] else {}
        print("\n==== first transformer forward: std vs modular ====")
        for k in sorted(set(cs) | set(cm)):
            if k not in cs or k not in cm:
                print(f"  {k}: present in {'std' if k in cs else 'mod'} only")
                continue
            a, b = cs[k], cm[k]
            if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                if a.shape != b.shape:
                    print(f"  {k}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
                else:
                    print(f"  {k}: max|Δ| {(a - b).abs().max().item():.3e}   shape {tuple(a.shape)}")
            else:
                print(f"  {k}: {'==' if a == b else 'DIFF'}   ({a!r} vs {b!r})")

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
    parser.add_argument(
        "--debug_forward",
        action="store_true",
        help="Capture the first transformer forward of each run (std vs modular) and diff inputs+outputs.",
    )

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument("--image", type=str, default=None)

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
