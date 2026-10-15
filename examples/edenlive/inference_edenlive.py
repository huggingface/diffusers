# Copyright 2026 The HuggingFace Team. All rights reserved.
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

"""Offline EdenLive inference, optionally using Ulysses and FSDP2 through torchrun."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from diffusers import ContextParallelConfig, EdenLivePipeline
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


def configure_parallelism(transformer, device, world_size, use_fsdp):
    if world_size == 1:
        transformer.to(device)
        return
    if world_size not in (2, 4):
        raise ValueError("This example supports 1, 2, or 4 processes.")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    mesh = init_device_mesh("cuda", (1, world_size), mesh_dim_names=("ring", "ulysses"))
    transformer.enable_parallelism(config=ContextParallelConfig(ring_degree=1, ulysses_degree=world_size, mesh=mesh))
    if use_fsdp:
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=False,
        )
        for block in transformer.blocks:
            fully_shard(block, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)
        fully_shard(transformer, mesh=mesh["ulysses"], mp_policy=policy, reshard_after_forward=True)

        def native_timestep(module, args, kwargs):
            # Match native FSDP time quantization without copying mutable cache dataclasses.
            if "timestep" in kwargs:
                kwargs = {**kwargs, "timestep": kwargs["timestep"].float().to(torch.bfloat16)}
            elif len(args) > 1:
                args = (args[0], args[1].float().to(torch.bfloat16), *args[2:])
            return args, kwargs

        transformer.register_forward_pre_hook(native_timestep, with_kwargs=True, prepend=True)
    else:
        transformer.to(device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="VAST-AI/EdenLive", help="EdenLive release directory or Hub repository.")
    parser.add_argument("--profile", choices=("5b-bidir", "14b-bidir", "5b-causal", "14b-causal"), required=True)
    parser.add_argument("--full-vae", action="store_true", help="Decode with the full Wan VAE instead of LightTAE.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--text-timeline", type=Path, help="JSON timeline for 5B bidirectional inference.")
    parser.add_argument("--chunk-prompts", type=Path, help="JSON list of prompts for chunk-causal inference.")
    parser.add_argument("--output", type=Path, default=Path("edenlive.mp4"))
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--fps", type=float)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attention-backend")
    parser.add_argument("--no-fsdp", action="store_true", help="Use replicated DiT weights with context parallelism.")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--vae-tiling", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size not in (1, 2, 4):
        raise ValueError("Launch this example with 1, 2, or 4 processes.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        loading_kwargs = {"decoder": None} if args.full_vae else {}
        pipeline = EdenLivePipeline.from_pretrained(
            args.model,
            profile=args.profile,
            local_files_only=args.local_files_only,
            **loading_kwargs,
        )
        if args.attention_backend is not None:
            pipeline.transformer.set_attention_backend(args.attention_backend)
        configure_parallelism(pipeline.transformer, device, world_size, not args.no_fsdp)
        for name, component in pipeline.components.items():
            if name != "transformer" and isinstance(component, torch.nn.Module):
                component.to(device=device)
        if args.vae_tiling:
            pipeline.vae.enable_tiling()
        pipeline.set_progress_bar_config(disable=rank != 0)
        is_5b = args.profile.startswith("5b-")
        args.height = args.height if args.height is not None else (704 if is_5b else 480)
        args.width = args.width if args.width is not None else (1280 if is_5b else 832)
        args.fps = args.fps if args.fps is not None else (24 if is_5b else 16)
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
        for name in ("text_timeline", "chunk_prompts"):
            path = getattr(args, name)
            if path is not None:
                inputs[name] = json.loads(path.read_text())
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        frames = pipeline(**inputs).frames[0]
        torch.cuda.synchronize(device)
        metrics = {
            "rank": rank,
            "seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        all_metrics = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(all_metrics, metrics)
        else:
            all_metrics[0] = metrics
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(args.output), fps=args.fps)
            report = {
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                },
                "pipeline": pipeline.__class__.__name__,
                "video": video_info,
                "world_size": world_size,
                "fsdp2": world_size > 1 and not args.no_fsdp,
                "metrics": all_metrics,
            }
            args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
