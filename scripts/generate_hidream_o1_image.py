# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoProcessor

from diffusers import HiDreamO1ImagePipeline, HiDreamO1Transformer2DModel, UniPCMultistepScheduler


DEV_TIMESTEPS = [
    999,
    987,
    974,
    960,
    945,
    929,
    913,
    895,
    877,
    857,
    836,
    814,
    790,
    764,
    737,
    707,
    675,
    640,
    602,
    560,
    515,
    464,
    409,
    347,
    278,
    199,
    110,
    8,
]


def parse_args():
    parser = argparse.ArgumentParser("Generate an image with HiDream-O1")
    parser.add_argument("--model_path", default="HiDream-ai/HiDream-O1-Image")
    parser.add_argument(
        "--prompt",
        default=(
            "A cinematic portrait of a glass astronaut standing in a neon-lit botanical garden, "
            "highly detailed, sharp focus, natural skin tones, 35mm film still."
        ),
    )
    parser.add_argument("--output_image", default="hidream_o1_output.png")
    parser.add_argument("--height", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--noise_scale_start", type=float, default=8.0)
    parser.add_argument("--noise_scale_end", type=float, default=None)
    parser.add_argument("--noise_clip_std", type=float, default=0.0)
    parser.add_argument(
        "--dev_defaults",
        action="store_true",
        help="Use the public dev checkpoint generation defaults: 28 steps, no guidance, shift 1.0, and dev timesteps.",
    )
    parser.add_argument("--torch_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device_map passed to HiDreamO1Transformer2DModel.from_pretrained, for example `cuda` or `auto`.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--use_resolution_binning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Snap the requested size to the official predefined high-resolution buckets.",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_name: str):
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def main():
    args = parse_args()
    torch_dtype = get_torch_dtype(args.torch_dtype)

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=args.local_files_only)
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": args.local_files_only,
    }
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    transformer = HiDreamO1Transformer2DModel.from_pretrained(args.model_path, **load_kwargs).eval()
    pipe = HiDreamO1ImagePipeline(
        processor=processor,
        transformer=transformer,
        scheduler=UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=args.shift,
        ),
    )
    if args.device_map is None:
        pipe.to(args.device)

    timesteps = None
    num_inference_steps = args.num_inference_steps
    guidance_scale = args.guidance_scale
    shift = args.shift
    noise_scale_start = args.noise_scale_start
    noise_scale_end = args.noise_scale_end
    noise_clip_std = args.noise_clip_std

    if args.dev_defaults:
        timesteps = DEV_TIMESTEPS
        num_inference_steps = len(DEV_TIMESTEPS)
        guidance_scale = 0.0
        shift = 1.0
        noise_scale_start = 7.5
        noise_scale_end = 7.5
        noise_clip_std = 2.5

    generator_device = args.device if args.device_map is None else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        timesteps=timesteps,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        noise_clip_std=noise_clip_std,
        use_resolution_binning=args.use_resolution_binning,
        generator=generator,
    ).images[0]

    output_dir = os.path.dirname(os.path.abspath(args.output_image))
    os.makedirs(output_dir, exist_ok=True)
    image.save(args.output_image)
    print(f"Saved image to {args.output_image}")


if __name__ == "__main__":
    main()
