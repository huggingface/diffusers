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

import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from diffusers import (
    ComponentsManager,
    MagiClassifierFreeGuidance,
    MagiImageToVideoBlocks,
    MagiVideoToVideoBlocks,
    ModularPipeline,
)
from diffusers.utils import export_to_video


def main():
    parser = argparse.ArgumentParser(description="Run MAGI base generation using an official example config.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", default="Good Boy")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save-latents", action="store_true")
    prefix = parser.add_mutually_exclusive_group()
    prefix.add_argument("--image", type=Path)
    prefix.add_argument("--video", type=Path)
    args = parser.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("Choose a new or empty output directory.")
    config = json.loads(args.config.read_text())
    if config["engine_config"]["distill"] or config["engine_config"].get("fp8_quant", False):
        parser.error("This example supports non-quantized base checkpoints only.")
    runtime = config["runtime_config"]
    if runtime["cfg_number"] != 3:
        parser.error("MAGI base requires three-way guidance.")
    height = runtime["video_size_h"] if args.height is None else args.height
    width = runtime["video_size_w"] if args.width is None else args.width
    num_frames = runtime["num_frames"] if args.num_frames is None else args.num_frames
    seed = runtime["seed"] if args.seed is None else args.seed
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=args.device)
    workflow = "i2v" if args.image else "v2v" if args.video else "t2v"
    pixels = {}
    if args.image or args.video:
        filters = f"scale={width}:{height}"
        if args.video:
            filters = f"fps={runtime['fps']}," + filters
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(args.image or args.video),
                "-vf",
                filters,
                "-frames:v",
                "1" if args.image else "32",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        frames = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, height, width, 3).copy()
        if len(frames) == 0:
            raise ValueError("The prefix contains no decodable frames.")
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        pixels = {"image": tensor[:, :, 0]} if args.image else {"video": tensor}
        blocks = MagiImageToVideoBlocks() if args.image else MagiVideoToVideoBlocks()
        pipe = blocks.init_pipeline(args.model, components_manager=manager)
    else:
        pipe = ModularPipeline.from_pretrained(args.model, components_manager=manager)
    pipe.load_components(dtype={"default": torch.float32, "transformer": torch.bfloat16, "vae": torch.bfloat16})
    pipe.update_components(
        guider=MagiClassifierFreeGuidance(
            timestep_thresholds=runtime["cfg_t_range"],
            prefix_scales=runtime["prev_chunk_scales"],
            text_scales=runtime["text_scales"],
        )
    )
    if pipe.config.latent_scaling_factor != runtime["scale_factor"]:
        raise ValueError("The checkpoint and official config have different latent scaling factors.")
    pipe.transformer.set_attention_backend("flash_varlen")
    pipe.vae.set_attention_backend("flash")
    pipe.vae.enable_tiling(tile_sample_min_length=runtime["fps"] // 2)
    args.output.mkdir(parents=True, exist_ok=True)
    call = {
        "prompt": args.prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": runtime["num_steps"],
        "chunk_width": runtime["chunk_width"],
        "window_size": runtime["window_size"],
        "noise2clean_kvrange": runtime["noise2clean_kvrange"],
        "clean_chunk_kvrange": runtime["clean_chunk_kvrange"],
        "clean_t": runtime["clean_t"],
        "cache_device": "cpu" if config["engine_config"].get("kv_offload", False) else None,
        "output_type": "pt",
    }
    metadata = {
        "model": args.model,
        "workflow": workflow,
        "prefix_path": str(args.image or args.video) if pixels else None,
        "prefix_frames": int(tensor.shape[2]) if pixels else 0,
        "official_config": str(args.config),
        "call": call,
        "seed": seed,
        "fps": runtime["fps"],
        "vae_tile_sample_min_length": runtime["fps"] // 2,
        "device": torch.cuda.get_device_name(args.device),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "huggingface-hub", "accelerate", "flash-attn")
        },
    }
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    started = time.perf_counter()
    count = 0

    def progress(module, inputs):
        nonlocal count
        count += 1
        if count % 24 == 1:
            print(f"Transformer call {count}, elapsed {time.perf_counter() - started:.1f}s", flush=True)

    handle = pipe.transformer.register_forward_pre_hook(progress)
    torch.cuda.reset_peak_memory_stats(args.device)
    try:
        result = pipe(
            **call,
            **pixels,
            generator=torch.Generator(args.device).manual_seed(seed),
            output=["videos", "latents", "completed_chunks"] + (["conditioning_latents"] if pixels else []),
        )
    except Exception as error:
        metadata.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        (args.output / "failure.json").write_text(json.dumps(metadata, indent=2))
        raise
    finally:
        handle.remove()
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    video = result["videos"].cpu()
    assert video.isfinite().all() and result["latents"].isfinite().all()
    assert video.min() >= 0 and video.max() <= 1
    assert video.shape[0] == 1 and video.shape[2:] == (3, height, width)
    if args.save_latents:
        torch.save(result["latents"].cpu(), args.output / "latents.pt")
        if pixels:
            torch.save(result["conditioning_latents"].cpu(), args.output / "conditioning_latents.pt")
            torch.save(pixels, args.output / "prefix_pixels.pt")
    video_path = args.output / f"output_{workflow}.mp4"
    export_to_video(video[0].permute(0, 2, 3, 1).numpy(), str(video_path), fps=runtime["fps"])
    reader = imageio.get_reader(video_path)
    assert reader.count_frames() == video.shape[1]
    reader.close()
    metadata.update(
        elapsed_seconds=elapsed,
        peak_allocated_gib=torch.cuda.max_memory_allocated(args.device) / 2**30,
        transformer_calls=count,
        completed_chunks=result["completed_chunks"],
        shape=list(video.shape),
        video_std=video.std().item(),
        mean_frame_difference=(video[:, 1:] - video[:, :-1]).abs().mean().item(),
        finite=True,
    )
    (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
