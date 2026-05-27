#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal smoke-test runner for the Cosmos3 diffusers pipeline.

Canonical examples live in the docs page at
``docs/source/en/api/pipelines/cosmos3.md`` — copy from there for production use.
This script exists to exercise the full load → encode → denoise → decode path
during development.

Text-to-image:
    python inference_cosmos3.py --prompt "A robot in a lab." --num-frames 1

Text-to-video:
    python inference_cosmos3.py --prompt "A waterfall in a forest."

Image-to-video:
    python inference_cosmos3.py --prompt "..." --vision-path /path/to/image.jpg

Text-to-video-with-sound (requires a sound-capable checkpoint):
    python inference_cosmos3.py --prompt "..." --enable-sound
"""

import argparse
import pathlib

import torch
from huggingface_hub import snapshot_download

from diffusers import Cosmos3OmniPipeline
from diffusers.utils import encode_video, export_to_video, load_image


HF_REPOS = {
    "nano": "nvidia/Cosmos3-Nano",
    "super": "nvidia/Cosmos3-Super",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True, help="Text prompt.")
    parser.add_argument(
        "--model",
        choices=sorted(HF_REPOS),
        default="nano",
        help="Which Cosmos3 checkpoint to load (maps to the corresponding nvidia/Cosmos3-* repo).",
    )
    parser.add_argument(
        "--vision-path",
        default=None,
        help="Optional URL or local path for an image-conditioning frame (enables image-to-video).",
    )
    parser.add_argument("--output", default=".", help="Directory to save generated video/image/audio files.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=189,
        help="Number of frames to generate. Use 1 for text-to-image; defaults to 189 for video (≈ 7.9s @ 24 FPS).",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument(
        "--enable-sound",
        action="store_true",
        default=False,
        help="Generate sound alongside video (requires a sound-capable checkpoint).",
    )
    parser.add_argument(
        "--no-duration-template",
        dest="add_duration_template",
        action="store_false",
        default=True,
        help="Skip the duration metadata sentence appended to the prompt and negative prompt (video only).",
    )
    parser.add_argument(
        "--no-resolution-template",
        dest="add_resolution_template",
        action="store_false",
        default=True,
        help="Skip the resolution metadata sentence appended to the prompt and negative prompt.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        default=False,
        help="Disable the Cosmos Guardrail safety checker at pipeline construction (no checker instantiated).",
    )
    parser.add_argument(
        "--no-safety-check",
        action="store_true",
        default=False,
        help="Skip the Cosmos Guardrail text/video safety checks for this call (checker still constructed).",
    )
    args = parser.parse_args()

    hf_repo = HF_REPOS[args.model]
    print(f"Downloading pipeline from {hf_repo}")
    pipeline_path = pathlib.Path(snapshot_download(repo_id=hf_repo))
    print(f"Loading pipeline from {pipeline_path} …")
    pipeline = Cosmos3OmniPipeline.from_pretrained(
        str(pipeline_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        enable_safety_checker=not args.disable_safety_checker,
    )
    print("Pipeline loaded successfully.")

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.vision_path) if args.vision_path is not None else None

    result = pipeline(
        prompt=args.prompt,
        image=image,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        enable_sound=args.enable_sound,
        add_resolution_template=args.add_resolution_template,
        add_duration_template=args.add_duration_template,
        enable_safety_check=not args.no_safety_check,
    )

    if args.num_frames == 1:
        save_path = output_dir / "sample.jpg"
        result.video[0].save(save_path, format="JPEG", quality=85)
    else:
        save_path = output_dir / "sample.mp4"
        if result.sound is not None:
            assert pipeline.sound_tokenizer is not None
            encode_video(
                result.video,
                fps=int(args.fps),
                audio=result.sound,
                audio_sample_rate=pipeline.sound_tokenizer.config.sampling_rate,
                output_path=str(save_path),
            )
        else:
            # macro_block_size=1 allows arbitrary frame sizes (Cosmos3 outputs are not always divisible by 16).
            export_to_video(result.video, str(save_path), fps=int(args.fps), quality=10, macro_block_size=1)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
