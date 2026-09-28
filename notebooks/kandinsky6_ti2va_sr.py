#!/usr/bin/env python3
"""Kandinsky 6 T2VA -> SR, stacked.

Script version of kandinsky6_ti2va_sr.ipynb, for running on a headless server.

Follows the two-stage pattern used by LTX-2.5: load the T2VA and SR artifacts
independently, generate video and audio with T2VA, then pass the decoded video
and generated audio to SR directly (no intermediate video encode/decode). The
pipelines remain separate and can be loaded, offloaded, or replaced independently.

The production SR route is --resolution-scale 2.25 (1.125x pixel pre-upscale
followed by the x2 KVAE path).

Usage:
    python kandinsky6_ti2va_sr.py
    python kandinsky6_ti2va_sr.py --t2va-model /path/to/local/bundle --sr-model /path/to/local/sr_bundle

Authenticate with the Hub before running if a model repo is gated:
    huggingface-cli login
  or:
    export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

DEFAULT_T2VA_MODEL = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
DEFAULT_SR_MODEL = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t2va-model", default=DEFAULT_T2VA_MODEL, help="Hub repo id or local bundle path.")
    parser.add_argument("--sr-model", default=DEFAULT_SR_MODEL, help="Hub repo id or local bundle path.")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic shot of a small red sailboat crossing a calm blue lake at sunset, "
            "with gentle waves and natural ambient sound."
        ),
    )
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality, noisy audio")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument(
        "--num-frames", type=int, default=121, help="Must be 1 + 8k: Kandinsky6SRPipeline rejects other values."
    )
    parser.add_argument("--sample-fps", type=float, default=24.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--resolution-scale", type=float, default=2.25)
    parser.add_argument("--output", type=Path, default=Path("outputs/diffusers/kandinsky6_t2va_sr.mp4"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    t2va_pipe = Kandinsky6TI2VAPipeline.from_pretrained(args.t2va_model, torch_dtype=torch.bfloat16)
    t2va_pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)

    sr_pipe = Kandinsky6SRPipeline.from_pretrained(args.sr_model, torch_dtype=torch.bfloat16)
    sr_pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)

    print(f"T2VA artifact: {args.t2va_model}")
    print(f"SR artifact: {args.sr_model}")

    t2va_result = t2va_pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        sample_audio=True,
        output_type="pt",
    )
    if t2va_result.audio is None:
        raise RuntimeError("The T2VA pipeline returned no audio")
    print(f"T2VA frames: {tuple(t2va_result.frames.shape)}")
    print(f"T2VA audio streams: {len(t2va_result.audio)}")

    sr_result = sr_pipe(
        video=t2va_result.frames,
        resolution_scale=args.resolution_scale,
        output_type="pt",
    )
    print(f"SR frames: {tuple(sr_result.frames.shape)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    audio = torch.as_tensor(t2va_result.audio[0])[:, None].repeat(1, 2)
    encode_video(
        sr_result.frames[0].permute(1, 2, 3, 0),
        fps=int(args.sample_fps),
        output_path=str(args.output),
        audio=audio,
        audio_sample_rate=int(t2va_pipe.audio_sample_rate),
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
