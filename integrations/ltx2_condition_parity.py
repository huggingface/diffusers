"""
LTX-2.5 condition-to-video parity harness: modular `LTX2ConditionBlocks` vs. standard `LTX2ConditionPipeline`.

TEMPORARY / for-visibility only. This whole `integrations/` directory is meant to be committed
transiently while the modular LTX-2 integration is under review, and removed before the final merge.
It is NOT a pytest test and is not wired into CI — run it manually against a real checkpoint.

What it does
------------
Loads the checkpoint once, then runs the *same* condition-based generation through:
  1. the standard `LTX2ConditionPipeline` (`.__call__`), and
  2. the modular `LTX2ConditionBlocks` blockset (built with `init_pipeline()`),

reusing the *identical* loaded component objects for both (via `update_components`). Because the
weights are shared, any output difference isolates a difference in the *block logic*, not in loading.

Both runs:
  - use the same `LTX2VideoCondition` objects (deterministic synthetic gradient frames by default);
  - use `output_type="latent"` so we compare the denoised latents directly;
  - use a freshly-seeded generator with the same seed. Conditions are VAE-encoded with `argmax` (mode,
    no randomness), so both pipelines consume randomness in the same order: video noise, audio noise;
  - disable prompt enhancement (`enable_prompt_enhancement=False` on both pipelines);
  - skip the image-conditioning H.264 re-compression by default (`--crf 0`), so the comparison isolates
    the block logic rather than the PyAV codec round-trip. Pass `--crf -1` to let each pipeline resolve
    the model default (18 for LTX-2.5) and check parity through the re-compression too.

Cases worth running
-------------------
    # first-frame condition only (latent index 0) -- should track the i2v numbers
    python integrations/ltx2_condition_parity.py --condition 0 1.0 1

    # keyframe condition only (latent index > 0) -- exercises appended tokens, keyframe coords, and the
    # `mu`-from-full-sequence ordering
    python integrations/ltx2_condition_parity.py --condition 2 1.0 1

    # both together, plus a multi-frame (video) condition
    python integrations/ltx2_condition_parity.py --condition 0 1.0 1 --condition 2 0.5 9

    # through the CRF re-compression
    python integrations/ltx2_condition_parity.py --condition 0 1.0 1 --crf -1

    # no conditions at all -- the condition blockset should reduce to plain text-to-video
    python integrations/ltx2_condition_parity.py --no_conditions

Lower `--num_inference_steps` keeps the run fast; the fp32 gate holds at any step count (the bf16 gap
grows with steps -- see the DTYPE_TOLERANCES note for why fp32 is the authoritative gate).
"""

import argparse

import numpy as np
import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, LTX2ConditionPipeline
from diffusers.modular_pipelines.ltx2 import LTX2ConditionBlocks
from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT
from diffusers.utils import load_image


DEFAULT_PROMPT = "The fox turns its head and blinks slowly as snow begins to fall."

# Full guidance stack (CFG + spatio-temporal guidance + modality-isolation), matching real LTX-2.5 usage.
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


# Components shared between the standard and modular pipelines (everything LTX2ConditionBlocks needs).
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


def _condition_frames(width: int, height: int, num_frames: int, offset: int) -> Image.Image | list[Image.Image]:
    # Deterministic synthetic gradient — no network dependency. The content is irrelevant to parity (both
    # pipelines receive the exact same frames); `offset` just makes distinct conditions visibly distinct.
    frames = []
    for i in range(num_frames):
        yy, xx = np.mgrid[0:height, 0:width]
        r = ((xx + offset + i) / width * 255 % 255).astype(np.uint8)
        g = (yy / height * 255).astype(np.uint8)
        b = ((xx + yy) / (width + height) * 255).astype(np.uint8)
        frames.append(Image.fromarray(np.stack([r, g, b], axis=-1)))
    # A single-frame condition is passed as a bare PIL image, which is what triggers the CRF re-compression
    # path in `preprocess_conditions`; multi-frame conditions are passed as a list and are never re-compressed.
    return frames[0] if num_frames == 1 else frames


def _build_conditions(args) -> list[LTX2VideoCondition] | None:
    if args.no_conditions:
        return None
    crf = None if args.crf < 0 else args.crf
    conditions = []
    for offset, (index, strength, num_frames) in enumerate(args.condition):
        frames = (
            load_image(args.condition_image)
            if args.condition_image is not None and num_frames == 1
            else _condition_frames(args.width, args.height, num_frames, offset * 37)
        )
        conditions.append(LTX2VideoCondition(frames=frames, index=index, strength=strength, crf=crf))
    return conditions


# Parity tolerances. The modular denoiser runs each guidance pass as its own single-batch transformer forward,
# whereas the standard pipeline batches cond+uncond into one forward. The two are mathematically equivalent, but
# GPU matmul is not batch-invariant -- `cond` computed alone differs from the same `cond` inside a batch-of-2 --
# so the modular and standard latents differ by genuine kernel-reordering noise (amplified by the guidance deltas
# and accumulated over sampler steps), NOT a logic bug. We therefore do NOT gate on assert_close's fp32 defaults:
# rtol=1.3e-6 is effectively a *bitwise* bar, and relative error at near-zero-valued elements explodes without
# meaning. Instead we gate on two magnitude-aware statistics, keyed off the *run* dtype:
#   - mean abs diff relative to mean magnitude -- the bulk-agreement signal, and
#   - max abs diff -- a loose ceiling that still catches structured / systematic errors.
# fp32 is the authoritative gate. The bounds mirror the i2v harness: like i2v, the condition workflow carries a
# per-token masked video timestep and clean anchor tokens the denoised tokens attend to, which amplifies the
# batch-invariance noise well beyond the t2v case. For the tight bug-catching run, disable CFG on BOTH sides
# (`--guidance_scale 1.0 --audio_guidance_scale 1.0`): with no cond/uncond batching the two should agree to
# ~1e-6 mean-rel. bf16 is a close-but-not-bitwise sanity check.
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
    conditions = _build_conditions(args)
    if conditions is None:
        print("Running WITHOUT conditions (condition blockset should reduce to text-to-video).")
    else:
        for c in conditions:
            n = 1 if isinstance(c.frames, Image.Image) else len(c.frames)
            print(f"condition: latent index {c.index}, strength {c.strength}, {n} frame(s), crf {c.crf}")

    # 1. Load the standard pipeline once.
    print(f"Loading {args.model_path} ...")
    std = LTX2ConditionPipeline.from_pretrained(args.model_path, torch_dtype=args.dtype)
    # For now, disable shift_terminal so that num_inference_steps=1 doesn't produce nans.
    new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(std.scheduler.config, shift_terminal=None)
    std.scheduler = new_scheduler
    if args.cpu_offload:
        std.enable_model_cpu_offload(device=args.device)
    else:
        std.to(args.device)

    # `num_frames=None` asks both pipelines to auto-predict the duration via the `duration_head`.
    num_frames = None if args.predict_duration else args.num_frames
    # Non-guidance call kwargs shared by both pipelines. Guidance is delivered separately: `__call__` kwargs for
    # the standard pipeline, guider components for the modular one.
    common_kwargs = {
        "conditions": conditions,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": num_frames,
        "min_seconds": args.min_seconds,
        "max_seconds": args.max_seconds,
        "noise_scale": args.noise_scale,
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
    print("Running standard LTX2ConditionPipeline ...")

    # Optional: capture the first transformer forward of each run. Both pipelines share `std.transformer`, so
    # identical inputs must give identical outputs -- diffing the first forward's kwargs pinpoints which input
    # first diverges (or shows identical inputs => the divergence is in the combine/step, not the forward).
    forward_captures = {"std": [], "mod": []}
    phase = {"name": None}
    hook_handle = None
    if args.debug_forward:

        def _capture_hook(module, fwd_args, fwd_kwargs, output):
            name = phase["name"]
            if name is None or forward_captures[name]:
                return
            snap = {k: (v.detach().float().cpu() if isinstance(v, torch.Tensor) else v) for k, v in fwd_kwargs.items()}
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

    # 3. Build the modular condition pipeline reusing the SAME component objects (identical weights), and configure
    #    the guiders with the same settings (guidance is guider config in modular diffusers, not a call kwarg).
    print("Building modular LTX2ConditionBlocks pipeline (shared components) ...")
    mod = LTX2ConditionBlocks().init_pipeline()
    mod.update_components(**{name: getattr(std, name) for name in SHARED_COMPONENTS})
    if getattr(std, "duration_head", None) is not None:
        mod.update_components(duration_head=std.duration_head)
    video_guider, audio_guider = _make_guiders(guidance)
    mod.update_components(guider=video_guider, audio_guider=audio_guider)

    # 4. Modular run — guidance comes from the guiders, so `GUIDANCE` is NOT passed here.
    print("Running modular LTX2ConditionBlocks ...")
    if args.debug_forward:
        phase["name"] = "mod"
    generator = torch.Generator(args.device).manual_seed(args.seed)
    state = mod(generator=generator, enable_prompt_enhancement=False, **common_kwargs)
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

    parser.add_argument(
        "--condition",
        nargs=3,
        action="append",
        metavar=("LATENT_INDEX", "STRENGTH", "NUM_FRAMES"),
        default=None,
        help="Repeatable. A frame condition at LATENT_INDEX with STRENGTH, built from NUM_FRAMES synthetic "
        "frames (1 => a bare PIL image, which is the CRF-re-compressed path). Defaults to `0 1.0 1`.",
    )
    parser.add_argument(
        "--no_conditions",
        action="store_true",
        help="Pass `conditions=None` to both pipelines (condition blockset reduces to text-to-video).",
    )
    parser.add_argument(
        "--condition_image",
        type=str,
        default=None,
        help="Optional real image (path or URL) used for single-frame conditions instead of the synthetic gradient.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=0,
        help="Image-conditioning H.264 CRF applied to every single-frame condition, for both pipelines. "
        "0 (default) skips re-compression; -1 resolves the model default (18 for LTX-2.5).",
    )

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="Initial noise level for the un-conditioned tokens. Default (None) resolves to sigmas[0], else 1.0.",
    )

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
        help="Parity gate: mean abs diff relative to mean magnitude. Dtype-aware default (fp32: 2e-2, bf16: 0.15).",
    )
    parser.add_argument(
        "--max_abs_tol",
        type=float,
        default=None,
        help="Parity gate: max abs diff ceiling. Dtype-aware default (fp32: 1.5e-1, bf16: 0.5).",
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
    args.condition = [(int(i), float(s), int(n)) for i, s, n in (args.condition or [["0", "1.0", "1"]])]

    default_mean_rel_tol, default_max_abs_tol = DTYPE_TOLERANCES[args.dtype]
    if args.mean_rel_tol is None:
        args.mean_rel_tol = default_mean_rel_tol
    if args.max_abs_tol is None:
        args.max_abs_tol = default_max_abs_tol

    main(args)
