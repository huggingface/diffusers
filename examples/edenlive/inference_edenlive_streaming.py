# Copyright 2026 The VAST-AI Team. All rights reserved.
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

"""Generate successive EdenLive chunks from ordered, already sampled proxy images."""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

from diffusers import EdenLivePipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="VAST-AI/EdenLive")
    parser.add_argument("--profile", choices=("5b-causal", "14b-causal"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--proxy-frames", nargs="+", required=True, help="Ordered, already sampled RGB image paths.")
    parser.add_argument("--chunk-prompts", type=Path, help="JSON file containing one prompt string per chunk.")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, help="Playback rate; defaults to 16 for 14b-causal and 24 for 5b-causal.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.fps = args.fps if args.fps is not None else (16 if args.profile == "14b-causal" else 24)
    if len(args.proxy_frames) % 3:
        parser.error("--proxy-frames must contain a multiple of three image paths.")
    if args.fps <= 0:
        parser.error("--fps must be positive.")
    prompts = None
    if args.chunk_prompts is not None:
        prompts = json.loads(args.chunk_prompts.read_text())
        if (
            not isinstance(prompts, list)
            or len(prompts) != len(args.proxy_frames) // 3
            or not all(isinstance(prompt, str) for prompt in prompts)
        ):
            parser.error("--chunk-prompts must contain one string per chunk.")
    pipe = EdenLivePipeline.from_pretrained(
        args.model, profile=args.profile, local_files_only=args.local_files_only
    ).to("cuda")
    with Image.open(args.image) as image:
        state = pipe.create_stream_state(
            image=image.convert("RGB"),
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="np",
        )
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(args.output), mode="w") as container:
            stream = container.add_stream("libx264", rate=args.fps)
            stream.height, stream.width = state.height, state.width
            stream.pix_fmt = "yuv420p"
            for index in range(0, len(args.proxy_frames), 3):
                frames = []
                for path in args.proxy_frames[index : index + 3]:
                    with Image.open(path) as frame:
                        frames.append(frame.convert("RGB"))
                output = pipe(
                    stream_state=state, proxy_frames=frames, prompt=None if prompts is None else prompts[index // 3]
                )
                for frame in output.frames[0]:
                    pixels = np.clip(frame * 255, 0, 255).round().astype(np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                        container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    finally:
        pipe.close_stream(state)


if __name__ == "__main__":
    main()
