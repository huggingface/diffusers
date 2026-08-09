"""
LTX-2.5 in-context (IC-LoRA) parity harness: modular `LTX2InContextBlocks` vs. standard `LTX2InContextPipeline`.

TEMPORARY / for-visibility only. This whole `integrations/` directory is meant to be committed
transiently while the modular LTX-2 integration is under review, and removed before the final merge.
It is NOT a pytest test and is not wired into CI — run it manually against a real checkpoint.

What it does
------------
Loads the checkpoint once, then runs the *same* in-context generation through:
  1. the standard `LTX2InContextPipeline` (`.__call__`), and
  2. the modular `LTX2InContextBlocks` blockset (built with `init_pipeline()`),

reusing the *identical* loaded component objects for both (via `update_components`). Because the
weights are shared, any output difference isolates a difference in the *block logic*, not in loading.
No IC-LoRA is loaded: parity is about the block logic, and both sides run the same base transformer.

Both runs use the same `LTX2ReferenceCondition` (and optional `LTX2VideoCondition`) objects, built from
deterministic synthetic frames; `output_type="latent"`; the same seed; and prompt enhancement disabled.

Cases worth running
-------------------
    # reference only, unmasked attention -- the core IC path
    python integrations/ltx2_in_context_parity.py --reference 1.0 9

    # reference at partial strength (exercises the x0 re-pin that the `has_conditions` gate used to skip)
    python integrations/ltx2_in_context_parity.py --reference 0.5 9

    # masked attention, scalar strength -> builds the video self-attention mask
    python integrations/ltx2_in_context_parity.py --reference 1.0 9 --conditioning_attention_strength 0.5

    # masked attention from a pixel-space mask
    python integrations/ltx2_in_context_parity.py --reference 1.0 9 --attention_mask_video

    # downscaled reference (coords scaled into the target space)
    python integrations/ltx2_in_context_parity.py --reference 1.0 9 --reference_downscale_factor 2

    # references alongside frame conditions, including a keyframe
    python integrations/ltx2_in_context_parity.py --reference 1.0 9 --condition 0 1.0 1 --condition 2 0.5 9

    # NO reference conditions -- IC-LoRAs that carry their behavior in the adapter weights (camera control,
    # style, ...) take no reference video; this is the shape of the standard pipeline's own docstring example
    python integrations/ltx2_in_context_parity.py --no_references --condition 0 1.0 1

    # two references
    python integrations/ltx2_in_context_parity.py --reference 1.0 9 --reference 0.8 9
"""

import argparse

import numpy as np
import torch
from PIL import Image

from diffusers import FlowMatchEulerDiscreteScheduler, LTX2InContextPipeline
from diffusers.modular_pipelines.ltx2 import LTX2InContextBlocks
from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT


DEFAULT_PROMPT = "The fox turns its head and blinks slowly as snow begins to fall."

# Full guidance stack (CFG + spatio-temporal guidance + modality-isolation), matching real LTX-2.5 usage.
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
USE_CROSS_TIMESTEP = True


def _resolve_guidance(args) -> dict:
    """Apply the CLI guidance overrides on top of the GUIDANCE defaults. The SAME resolved dict drives BOTH the
    standard pipeline (as `__call__` kwargs) and the modular guiders."""
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
    )
    return video_guider, audio_guider


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


def _frames(width: int, height: int, num_frames: int, offset: int) -> Image.Image | list[Image.Image]:
    # Deterministic synthetic gradient — content is irrelevant to parity (both pipelines get the same frames).
    frames = []
    for i in range(num_frames):
        yy, xx = np.mgrid[0:height, 0:width]
        r = ((xx + offset + i) / width * 255 % 255).astype(np.uint8)
        g = (yy / height * 255).astype(np.uint8)
        b = ((xx + yy) / (width + height) * 255).astype(np.uint8)
        frames.append(Image.fromarray(np.stack([r, g, b], axis=-1)))
    return frames[0] if num_frames == 1 else frames


def _attention_mask_video(width: int, height: int, num_frames: int, device: str) -> torch.Tensor:
    # Pixel-space mask of shape (1, 1, F, H, W) in [0, 1]: a horizontal ramp, constant in time.
    yy, xx = np.mgrid[0:height, 0:width]
    plane = torch.from_numpy((xx / max(width - 1, 1)).astype(np.float32))
    return plane.expand(1, 1, num_frames, height, width).contiguous().to(device)


DTYPE_TOLERANCES = {
    # dtype: (mean_rel_tol, max_abs_tol) -- see ltx2_condition_parity.py for the rationale. Same shape of argument:
    # the modular denoiser runs each guidance pass as its own single-batch forward while the standard pipeline
    # batches cond+uncond, and GPU matmul is not batch-invariant. For the tight bug-catching run disable CFG on
    # BOTH sides (`--guidance_scale 1.0 --audio_guidance_scale 1.0`).
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
    ref_height = args.height // args.reference_downscale_factor
    ref_width = args.width // args.reference_downscale_factor
    reference_conditions = None
    if not args.no_references:
        reference_conditions = [
            LTX2ReferenceCondition(frames=_frames(ref_width, ref_height, n, i * 53), strength=s)
            for i, (s, n) in enumerate(args.reference)
        ]
        for c in reference_conditions:
            n = 1 if isinstance(c.frames, Image.Image) else len(c.frames)
            print(f"reference: strength {c.strength}, {n} frame(s) at {ref_width}x{ref_height}")
    else:
        print("Running WITHOUT reference conditions (reference encoder and attention-mask step are skipped).")

    conditions = None
    if args.condition:
        crf = None if args.crf < 0 else args.crf
        conditions = [
            LTX2VideoCondition(frames=_frames(args.width, args.height, n, i * 37), index=idx, strength=s, crf=crf)
            for i, (idx, s, n) in enumerate(args.condition)
        ]
        for c in conditions:
            n = 1 if isinstance(c.frames, Image.Image) else len(c.frames)
            print(f"condition: latent index {c.index}, strength {c.strength}, {n} frame(s), crf {c.crf}")

    attention_mask = None
    if args.attention_mask_video:
        attention_mask = _attention_mask_video(ref_width, ref_height, args.num_frames, args.device)
        print(f"attention mask: {tuple(attention_mask.shape)}")

    # 1. Load the standard pipeline once.
    print(f"Loading {args.model_path} ...")
    std = LTX2InContextPipeline.from_pretrained(args.model_path, torch_dtype=args.dtype)
    new_scheduler = FlowMatchEulerDiscreteScheduler.from_config(std.scheduler.config, shift_terminal=None)
    std.scheduler = new_scheduler
    if args.cpu_offload:
        std.enable_model_cpu_offload(device=args.device)
    else:
        std.to(args.device)

    common_kwargs = {
        "reference_conditions": reference_conditions,
        "conditions": conditions,
        "reference_downscale_factor": args.reference_downscale_factor,
        "conditioning_attention_strength": args.conditioning_attention_strength,
        "conditioning_attention_mask": attention_mask,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": args.num_frames,
        "noise_scale": args.noise_scale,
        "frame_rate": args.frame_rate,
        "num_inference_steps": args.num_inference_steps,
        "max_sequence_length": args.max_sequence_length,
        "num_videos_per_prompt": args.num_videos_per_prompt,
        "use_cross_timestep": USE_CROSS_TIMESTEP,
        "output_type": "latent",
    }

    guidance = _resolve_guidance(args)

    # 2. Standard run — guidance scales passed as `__call__` kwargs.
    print("Running standard LTX2InContextPipeline ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    video_std, audio_std = std(
        generator=generator,
        enable_prompt_enhancement=False,
        return_dict=False,
        **common_kwargs,
        **guidance,
    )

    # 3. Build the modular in-context pipeline reusing the SAME component objects.
    print("Building modular LTX2InContextBlocks pipeline (shared components) ...")
    mod = LTX2InContextBlocks().init_pipeline()
    mod.update_components(**{name: getattr(std, name) for name in SHARED_COMPONENTS})
    video_guider, audio_guider = _make_guiders(guidance)
    mod.update_components(guider=video_guider, audio_guider=audio_guider)

    # 4. Modular run — guidance comes from the guiders.
    print("Running modular LTX2InContextBlocks ...")
    generator = torch.Generator(args.device).manual_seed(args.seed)
    state = mod(generator=generator, enable_prompt_enhancement=False, **common_kwargs)
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

    parser.add_argument("--model_path", type=str, default=None, help="Model path loadable by from_pretrained")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_offload", action="store_true")

    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default=None)

    parser.add_argument(
        "--reference",
        nargs=2,
        action="append",
        metavar=("STRENGTH", "NUM_FRAMES"),
        default=None,
        help="Repeatable. An IC-LoRA reference video of NUM_FRAMES synthetic frames at STRENGTH. Defaults to `1.0 9`.",
    )
    parser.add_argument(
        "--no_references",
        action="store_true",
        help="Pass `reference_conditions=None` to both pipelines (IC-LoRAs that take no reference video).",
    )
    parser.add_argument(
        "--condition",
        nargs=3,
        action="append",
        metavar=("LATENT_INDEX", "STRENGTH", "NUM_FRAMES"),
        default=None,
        help="Repeatable. Optional frame condition alongside the references.",
    )
    parser.add_argument("--crf", type=int, default=0, help="CRF for single-frame conditions; -1 = model default.")
    parser.add_argument("--reference_downscale_factor", type=int, default=1)
    parser.add_argument("--conditioning_attention_strength", type=float, default=1.0)
    parser.add_argument(
        "--attention_mask_video",
        action="store_true",
        help="Supply a synthetic pixel-space `conditioning_attention_mask` (a horizontal ramp).",
    )

    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--frame_rate", type=float, default=24.0)
    parser.add_argument("--num_inference_steps", type=int, default=6)
    parser.add_argument("--max_sequence_length", type=int, default=1024)
    parser.add_argument("--num_videos_per_prompt", type=int, default=1)
    parser.add_argument("--noise_scale", type=float, default=None)

    parser.add_argument("--guidance_scale", type=float, default=None, help="Video CFG guidance scale")
    parser.add_argument("--stg_scale", type=float, default=None, help="Video STG guidance scale")
    parser.add_argument("--mod_scale", type=float, default=None, help="Video modality isolation guidance scale")
    parser.add_argument("--guidance_rescale", type=float, default=None, help="Video guidance rescale")
    parser.add_argument("--audio_guidance_scale", type=float, default=None, help="Audio CFG guidance scale")
    parser.add_argument("--audio_stg_scale", type=float, default=None, help="Audio STG guidance scale")
    parser.add_argument("--audio_mod_scale", type=float, default=None, help="Audio modality isolation guidance scale")
    parser.add_argument("--audio_guidance_rescale", type=float, default=None, help="Audio guidance rescale")

    parser.add_argument("--mean_rel_tol", type=float, default=None)
    parser.add_argument("--max_abs_tol", type=float, default=None)
    parser.add_argument("--check_tensor_stats", action="store_true")

    args = parser.parse_args()

    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    args.prompt = args.prompt or DEFAULT_PROMPT
    args.negative_prompt = args.negative_prompt or DEFAULT_NEGATIVE_PROMPT
    args.reference = [(float(s), int(n)) for s, n in (args.reference or [["1.0", "9"]])]
    args.condition = [(int(i), float(s), int(n)) for i, s, n in (args.condition or [])]

    default_mean_rel_tol, default_max_abs_tol = DTYPE_TOLERANCES[args.dtype]
    if args.mean_rel_tol is None:
        args.mean_rel_tol = default_mean_rel_tol
    if args.max_abs_tol is None:
        args.max_abs_tol = default_max_abs_tol

    main(args)
