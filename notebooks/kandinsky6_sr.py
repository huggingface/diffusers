#!/usr/bin/env python3
"""Kandinsky 6 video super-resolution.

Script version of kandinsky6_sr.ipynb, for running on a headless server.

The pipeline itself only accepts/returns tensors (no file I/O): this script
does the mp4 <-> tensor conversion with the shared `diffusers.utils`
video helpers, the same way a user integrating the pipeline into their own
code would.

Usage:
    python kandinsky6_sr.py --input-video assets/vsr_input.mp4
    python kandinsky6_sr.py --input-video a.mp4 b.mp4 --gpu-id 1 --resolution-scale 2.25

Authenticate with the Hub before running if the model repo is gated:
    huggingface-cli login
  or:
    export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
from os import pipe
from pathlib import Path

import numpy as np
import torch

from diffusers import Kandinsky6SRPipeline
from diffusers.utils import encode_video, load_video

DEFAULT_MODEL = "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--input-video",
        type=Path,
        nargs="+",
        required=True,
        help="One or more input video paths. A single path is replicated --batch-size times.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Only used when --input-video has one path.")
    parser.add_argument("--resolution-scale", type=float, default=2.25)
    parser.add_argument("--seed", type=int, default=1137)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffusers/k6_vsr"))
    return parser.parse_args()


def load_video_tensor(path: Path) -> tuple[torch.Tensor, float]:
    """Decode ``path`` into a ``[3, T, H, W]`` uint8 tensor, aligned to ``1 + 8k`` frames."""
    frames, fps = load_video(str(path), return_fps=True)
    video = torch.from_numpy(np.stack([np.array(frame) for frame in frames])).permute(3, 0, 1, 2)
    aligned = 1 + 8 * ((video.shape[1] - 1) // 8)
    if aligned == 0:
        raise SystemExit(f"{path}: no readable frames")
    return video[:, :aligned], fps


def main() -> None:
    args = parse_args()

    input_videos = args.input_video
    for path in input_videos:
        if not path.exists():
            raise SystemExit(f"input video not found: {path}")
    if len(input_videos) == 1 and args.batch_size > 1:
        input_videos = input_videos * args.batch_size
    batch_size = len(input_videos)

    loaded = [load_video_tensor(path) for path in input_videos]
    videos, fps_values = [item[0] for item in loaded], [item[1] for item in loaded]
    if any(video.shape != videos[0].shape for video in videos[1:]):
        raise SystemExit("all --input-video clips must share resolution and frame count")
    video_batch = torch.stack(videos)

    pipe = Kandinsky6SRPipeline.from_pretrained(args.model, dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)

    output_dir = args.output_dir / f"bs{batch_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / f"vsr_x{args.resolution_scale}_batchid{i}.mp4" for i in range(batch_size)]

    generator = torch.Generator(device=f"cuda:{args.gpu_id}").manual_seed(args.seed)

    # resolution_scale=2.25 means x1.125 pixel pre-upscale followed by the x2 KVAE path
    # (the production route).
    result = pipe(
        video=video_batch,
        resolution_scale=args.resolution_scale,
        output_type="pt",
        generator=generator,
    )
    print(f"frames: {tuple(result.frames.shape)}")
    for frames, output_path, fps in zip(result.frames, output_paths, fps_values, strict=True):
        encode_video(frames.permute(1, 2, 3, 0), fps=round(fps), output_path=str(output_path))
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
