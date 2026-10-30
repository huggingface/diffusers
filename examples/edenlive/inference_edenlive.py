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

"""Offline EdenLive inference on one GPU, with optional CPU offload."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (
    EdenLiveCausalPipeline,
    EdenLiveLightTAEDecoder,
    EdenLivePipeline,
)
from diffusers.utils import export_to_video


def load_video(path, fps, height, width, num_frames=None):
    import av

    if fps <= 0:
        raise ValueError("fps must be positive.")
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration is not None else None
        timestamps = np.asarray([float(frame.time) for frame in container.decode(video=0)])
    if len(timestamps) == 0:
        raise ValueError("The input video has no decoded frames.")
    timestamps -= timestamps[0]
    if duration is None:
        duration = timestamps[-1] + (np.median(np.diff(timestamps)) if len(timestamps) > 1 else 1 / fps)
    available = math.floor(duration * fps)
    count = 4 * ((available - 1) // 4) + 1 if num_frames is None else num_frames
    if count < 1 or (count - 1) % 4 or count > available:
        raise ValueError(f"num_frames must be 4k+1 and no larger than {available} frames at {fps} fps.")
    indices = np.searchsorted((timestamps[:-1] + timestamps[1:]) / 2, np.arange(count) / fps)
    frames = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if len(frames) == count:
                break
            if index != indices[len(frames)]:
                continue
            picture = frame.to_image().convert("RGB")
            crop_width = min(picture.width, round(picture.height * width / height))
            crop_height = min(picture.height, round(picture.width * height / width))
            left, top = (picture.width - crop_width) // 2, (picture.height - crop_height) // 2
            picture = picture.crop((left, top, left + crop_width, top + crop_height))
            picture = picture.resize((width, height), Image.Resampling.BILINEAR)
            while len(frames) < count and indices[len(frames)] == index:
                frames.append(picture)
    if len(frames) != count:
        raise ValueError(f"Decoded {len(frames)} of {count} requested frames.")
    return frames, {"source_frame_indices": indices.tolist(), "fps": fps, "num_frames": count}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Standard Diffusers model directory or Hub repository.")
    parser.add_argument("--causal", action="store_true", help="Load a causal model with EdenLiveCausalPipeline.")
    parser.add_argument(
        "--lighttae", action="store_true", help="Use the optional LightTAE decoder instead of the full Wan VAE."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--chunk-prompts", type=Path, help="JSON list of prompts for chunk-causal inference.")
    parser.add_argument("--output", type=Path, default=Path("edenlive.mp4"))
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, help="Noise seed; defaults to 42 for 14b-causal and 0 otherwise.")
    parser.add_argument("--attention-backend")
    parser.add_argument("--cpu-offload", action="store_true", help="Move models to the GPU only when needed.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    args = parser.parse_args()
    if args.chunk_prompts is not None and not args.causal:
        parser.error("--chunk-prompts requires --causal.")
    if args.prompt is None and args.chunk_prompts is None:
        parser.error("Provide --prompt or --chunk-prompts.")
    return args


def main():
    args = parse_args()
    device = torch.device("cuda")
    loading_kwargs = {"local_files_only": args.local_files_only}
    components = {"decoder": None}
    pipeline_class = EdenLiveCausalPipeline if args.causal else EdenLivePipeline
    if args.lighttae:
        components["decoder"] = EdenLiveLightTAEDecoder.from_pretrained(
            args.model, subfolder="decoder", torch_dtype=torch.bfloat16, **loading_kwargs
        )
    pipeline = pipeline_class.from_pretrained(
        args.model,
        torch_dtype={"default": torch.bfloat16, "proxy_encoder": torch.float32},
        **components,
        **loading_kwargs,
    )
    if args.attention_backend is not None:
        pipeline.transformer.set_attention_backend(args.attention_backend)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(device)
    if args.vae_tiling:
        pipeline.vae.enable_tiling()
    is_5b = pipeline.transformer.config.image_dim is None
    args.height = args.height if args.height is not None else (704 if is_5b else 480)
    args.width = args.width if args.width is not None else (1280 if is_5b else 832)
    args.fps = args.fps if args.fps is not None else (24 if is_5b else 16)
    args.seed = args.seed if args.seed is not None else (42 if args.causal and not is_5b else 0)
    video, video_info = load_video(args.video, args.fps, args.height, args.width, args.num_frames)
    inputs = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "image": Image.open(args.image).convert("RGB"),
        "video": video,
        "height": args.height,
        "width": args.width,
        "num_frames": len(video),
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        "output_type": "np",
    }
    for name in ("num_inference_steps", "guidance_scale"):
        if getattr(args, name) is not None:
            inputs[name] = getattr(args, name)
    if args.chunk_prompts is not None:
        inputs["chunk_prompts"] = json.loads(args.chunk_prompts.read_text())
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    frames = pipeline(**inputs).frames[0]
    torch.cuda.synchronize(device)
    metrics = {
        "seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=args.fps)
    report = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pipeline": pipeline.__class__.__name__,
        "video": video_info,
        "metrics": metrics,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
