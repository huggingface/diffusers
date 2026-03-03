#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
import os
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import pydantic
import torch
import tyro
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video, load_image, load_video


def arch_invariant_rand(shape, dtype, device, seed=None):
    rng = np.random.RandomState(seed)
    random_array = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """
    Where to save the outputs
    """

    model_id: str | None = "nvidia/Cosmos-Predict2.5-2B"
    """
    Which model repository to use. Use "nvidia/Cosmos-Predict2.5-14B" for 14B variant
    """

    revision: str | None = "diffusers/base/post-trained"
    """
    Which variant of the model to use. Defaults to the base post-trained model. Here is a list of valid variants are:
    - diffusers/base/pre-trained
    - diffusers/base/post-trained

    """

    input_path: Annotated[Path | None, tyro.conf.arg(aliases=("-i",))] = None
    """
    Path to the conditioning media (image or video) or to a JSON config file.

    If the path ends with .json the script will load prompt/media values from that file, this JSON file is expected to be the same file used for the example scripts.

    CLI arguments override JSON values when both are provided (overriding the conditioning input image or video must be done with `--override_visual_input <path>`).
    """

    override_visual_input: Path | None = None
    """
    Override media (image/video) path when `--input_path` points to a JSON asset.

    Useful when reusing JSON configs but swapping in a different conditioning file.
    """

    prompt: str | None = None
    """
    A string describing the prompt
    """

    prompt_path: Path | None = None
    """
    A text file describing the prompt, if provided will ignore prompt
    """

    num_output_frames: int = 93
    """
    Number of output frames. Use 1 for "2Image" mode and 93 for "2Video" mode.
    """

    negative_prompt: str | None = None
    """
    Negative prompt for cfg. If not provided, DEFAULT_NEGATIVE_PROMPT in Cosmos2_5_PredictBasePipeline will be applied.
    """

    negative_prompt_path: Path | None = None
    """
    Negative prompt file to use, if provided negative_prompt will be ignored. 
    """

    seed: int | None = 0
    """
    Seed for generation, if not provided no seed will be set
    """

    num_steps: int = 36
    """
    Number of steps to use
    """

    device: str = "cuda"
    """
    device to use
    """

    height: int = 704
    """
    Output height in pixels (must be divisible by 16).
    """

    width: int = 1280
    """
    Output width in pixels (must be divisible by 16).
    """

def load_inputs(args: Args):
    prompt = None
    negative_prompt = None
    resolved_input_path = None
    image = None
    video = None

    if args.prompt_path:
        prompt = args.prompt_path.open().read()
    elif args.prompt:
        prompt = args.prompt
    if args.negative_prompt_path:
        negative_prompt = args.negative_prompt_path.open().read()
    elif args.negative_prompt is not None:
        negative_prompt = args.negative_prompt

    if args.input_path and args.input_path.suffix.lower() == ".json":
        config = json.load(args.input_path.open())

        root_dir = args.input_path.parent
        if config.get("input_path") is not None:
            resolved_input_path = (root_dir / config["input_path"]).absolute()

        if prompt is None:
            if config.get("prompt_path") is not None:
                prompt = (root_dir / config["prompt_path"]).read_text().strip()
            elif config.get("prompt") is not None:
                prompt = config["prompt"]
        if negative_prompt is None and config.get("negative_prompt") is not None:
            negative_prompt = config["negative_prompt"]
    else:
        resolved_input_path = args.input_path

    if args.override_visual_input is not None:
        resolved_input_path = args.override_visual_input

    if resolved_input_path is not None:
        suffix = resolved_input_path.suffix.lower()
        if suffix == ".mp4":
            video = load_video(str(resolved_input_path))
        elif suffix in (".jpg", ".jpeg", ".png"):
            image = load_image(str(resolved_input_path))
        else:
            print(
                f"Unsupported input file extension '{resolved_input_path.suffix}'. "
                "Use .mp4 for video or .jpg/.jpeg/.png for images.",
                file=sys.stderr,
            )
            sys.exit(1)

    assert prompt is not None, "Prompt must be provided either through --prompt, --prompt_path, or the input JSON file."
    assert image is not None or video is not None, "Input must contain an image or a video."
    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "image": image,
        "video": video,
        "resolved_input_path": resolved_input_path,
    }


def format_output_path(output_path, num_output_frames):
    if num_output_frames == 1:
        if output_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            print(f"WARN: outputting to {str(output_path)}", file=sys.stderr)
            output_path = Path(f"{output_path}.jpg")
    else:
        if output_path.suffix.lower() != ".mp4":
            print(f"WARN: outputting to {str(output_path)}", file=sys.stderr)
            output_path = Path(f"{output_path}.mp4")

    os.makedirs(output_path.absolute().parent, exist_ok=True)
    return str(output_path)


def main(args: Args):

    class MockSafetyChecker:
        def to(self, *args, **kwargs):
            return self

        def check_text_safety(self, *args, **kwargs):
            return True

        def check_video_safety(self, video):
            return video

    inputs = load_inputs(args)

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        text_encoder_attn_implementation="flash_attention_2",
        safety_checker=MockSafetyChecker(),
    )

    latent_shape = pipe.get_latent_shape_cthw(args.height, args.width, args.num_output_frames)
    noises = arch_invariant_rand((1, *latent_shape), dtype=torch.float32, device=args.device, seed=args.seed)
    frames = pipe(
        image=inputs["image"],
        video=inputs["video"],
        prompt=inputs["prompt"],
        negative_prompt=inputs["negative_prompt"],
        num_frames=args.num_output_frames,
        num_inference_steps=args.num_steps,
        generator=torch.Generator().manual_seed(args.seed) if args.seed is not None else None,
        latents=noises, # optional argument to ensure architecture invariant generation
    ).frames[0]  # NOTE: batch_size == 1

    output_path = format_output_path(args.output_path, args.num_output_frames)
    if args.num_output_frames > 1:
        export_to_video(frames, output_path, fps=16)
    else:
        frames[0].save(output_path)


if __name__ == "__main__":
    args = tyro.cli(
        Args,
        description=__doc__,
        config=(tyro.conf.OmitArgPrefixes,),
    )
    main(args)
