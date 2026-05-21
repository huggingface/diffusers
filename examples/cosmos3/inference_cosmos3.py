#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Load the Cosmos3 diffusers pipeline from the HuggingFace Hub and run inference.

Text-to-video / image-to-video:
    python inference_cosmos3.py \
        --input inputs/omni/i2v.json

Text-to-video-sound (requires a sound-capable checkpoint):
    python inference_cosmos3.py \
        --input inputs/omni/t2v.json \
        --enable-sound

Text-to-image:
    python inference_cosmos3.py \
        --input inputs/omni/t2i.json \
"""

import argparse
import json
import pathlib

import torch
from huggingface_hub import snapshot_download

from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import Cosmos3OmniDiffusersPipeline, save_img_or_video, save_wav
from diffusers.utils import load_image


HF_REPO = "nvidia/Cosmos3-Nano"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="inputs/omni/i2v.json",
        help="Path to JSON input file with 'prompt' and optional 'vision_path'.",
    )
    parser.add_argument("--output", default=".", help="Directory to save generated video/image/audio files.")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to generate. If omitted, uses the JSON 'num_frames' "
        "field (defaults to 189, i.e. video).",
    )
    parser.add_argument(
        "--enable-sound",
        action="store_true",
        default=False,
        help="Generate sound alongside video (requires a sound-capable checkpoint).",
    )
    args = parser.parse_args()


    print(f"Downloading pipeline from {HF_REPO}")
    local_repo = snapshot_download(
        repo_id=HF_REPO,
    )
    pipeline_path = pathlib.Path(local_repo)
    print(f"Loading pipeline from {pipeline_path} …")
    pipeline = Cosmos3OmniDiffusersPipeline.from_pretrained(
        str(pipeline_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print("Pipeline loaded successfully.")

    # --- Load JSON input ---
    input_path = pathlib.Path(args.input)
    print(f"Loading input from {input_path} …")
    with open(input_path) as f:
        input_data = json.load(f)
    prompt = input_data["prompt"]
    vision_path = input_data.get("vision_path", None)
    num_frames = args.num_frames if args.num_frames is not None else int(input_data.get("num_frames", 189))

    output_dir = pathlib.Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(vision_path) if vision_path is not None else None

    result = pipeline(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        height=args.height,
        width=args.width,
        enable_sound=args.enable_sound,
    )

    ext = "jpg" if num_frames == 1 else "mp4"
    for i, frames in enumerate(result.video):
        save_path = str(output_dir / f"sample-{i}")
        save_img_or_video(frames, save_path)
        print(f"Saved: {save_path}.{ext}")

    if result.sound is not None:
        assert pipeline.sound_tokenizer is not None
        sample_rate = pipeline.sound_tokenizer.sample_rate
        for i, waveform in enumerate(result.sound):
            wav_path = output_dir / f"sample-{i}.wav"
            save_wav(waveform, wav_path, sample_rate)
            print(f"Saved: {wav_path}")


if __name__ == "__main__":
    main()
