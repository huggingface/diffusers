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

Video-to-video:
    python inference_cosmos3.py --prompt "..." --video-path /path/to/video.mp4

Text-to-video-with-sound (requires a sound-capable checkpoint):
    python inference_cosmos3.py --prompt "..." --enable-sound

Multi-GPU (any modality above): launch with torchrun and pass parallelism degrees.
``--tp-degree`` shards the weights (so large checkpoints fit), ``--cp-degree`` shards
the sequence (Ulysses, lower latency); ``--nproc_per_node`` must equal their product.
These reuse the helpers in the ``cosmos_{context,tensor}_parallel_inference.py`` examples.
    # TP=2 x CP=2 across 4 GPUs (Super):
    torchrun --nproc_per_node 4 inference_cosmos3.py --model super --tp-degree 2 --cp-degree 2 --prompt "..."
"""

import argparse
import json
import os
import pathlib
import sys
import urllib.request

import torch
from huggingface_hub import snapshot_download

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition, UniPCMultistepScheduler
from diffusers.utils import encode_video, export_to_video, load_image, load_video


# Multi-GPU helpers (context + tensor parallelism) live in the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cosmos_parallel import (  # noqa: E402
    enable_cosmos3_context_parallel,
    enable_cosmos3_flash_attention,
    enable_cosmos3_tensor_parallel,
)


HF_REPOS = {
    "nano": "nvidia/Cosmos3-Nano",
    "super": "nvidia/Cosmos3-Super",
}


def _load_action(path: str | None):
    if path is None:
        raise ValueError("--action-path is required for forward_dynamics mode.")
    if path.startswith(("http://", "https://")):
        with urllib.request.urlopen(path) as response:
            action = json.loads(response.read().decode("utf-8"))
    else:
        action = json.loads(pathlib.Path(path).read_text())
    tensor = torch.as_tensor(action, dtype=torch.float32)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Cosmos3 action must have shape [T, D], got {tuple(tensor.shape)}.")
    return tensor


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
        help="Optional URL or local path for an image-conditioning frame, or an action conditioning video.",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional URL or local path to a conditioning video for video-to-video generation.",
    )
    parser.add_argument(
        "--condition-frame-indexes-vision",
        default=None,
        help="Comma-separated latent frame indexes kept clean for video-to-video (default: 0,1).",
    )
    parser.add_argument(
        "--condition-video-keep",
        choices=["first", "last"],
        default="first",
        help="Take the video-to-video conditioning frames from the first or last of the source clip (default: first).",
    )
    parser.add_argument("--output", default=".", help="Directory to save generated video/image/audio files.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height in pixels (default 720). Ignored for action modes; use --resolution-tier instead.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width in pixels (default 1280). Ignored for action modes; use --resolution-tier instead.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=189,
        help="Number of frames to generate. Use 1 for text-to-image; defaults to 189 for video (≈ 7.9s @ 24 FPS).",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--guidance-scale", type=float, default=6.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num-inference-steps", type=int, default=35, help="Number of denoising steps.")
    parser.add_argument(
        "--flow-shift",
        type=float,
        default=None,
        help="Override the scheduler's flow-matching shift (UniPCMultistepScheduler.flow_shift).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for latent initialization.")
    parser.add_argument(
        "--tp-degree",
        type=int,
        default=1,
        help=(
            "Tensor-parallel degree: shard the model weights across this many GPUs (so large checkpoints "
            "fit). Must divide the query heads and KV heads. >1 requires launching with torchrun."
        ),
    )
    parser.add_argument(
        "--cp-degree",
        type=int,
        default=1,
        help=(
            "Context-parallel (Ulysses) degree: shard the sequence across this many GPUs (lower latency). "
            "--tp-degree * --cp-degree must equal --nproc_per_node, and must divide the query heads. "
            ">1 requires launching with torchrun."
        ),
    )
    parser.add_argument(
        "--enable-sound",
        action="store_true",
        default=False,
        help="Generate sound alongside video (requires a sound-capable checkpoint).",
    )
    parser.add_argument(
        "--action-mode",
        choices=["forward_dynamics", "inverse_dynamics", "policy"],
        default=None,
        help="Enable Cosmos3 action generation with a loaded conditioning video.",
    )
    parser.add_argument("--action-path", default=None, help="JSON action path for forward_dynamics mode.")
    parser.add_argument("--action-chunk-size", type=int, default=None, help="Number of action tokens to generate/use.")
    parser.add_argument("--domain-name", default=None, help="Cosmos3 action embodiment domain name.")
    parser.add_argument(
        "--view-point",
        choices=["ego_view", "third_person_view", "wrist_view", "concat_view"],
        default="ego_view",
        help="Camera perspective for the action caption's cinematography.framing field (default: ego_view).",
    )
    parser.add_argument(
        "--resolution-tier",
        type=int,
        default=480,
        choices=[256, 480, 704, 720],
        help=(
            "Action resolution tier (256/480/704/720). Selects the aspect bin / padded conditioning canvas, "
            "not the output frame size."
        ),
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

    tp, cp = args.tp_degree, args.cp_degree
    distributed = tp * cp > 1

    if distributed:
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        world = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        dev = torch.device("cuda", local_rank)
        if world != tp * cp:
            raise ValueError(
                f"--nproc_per_node ({world}) must equal --tp-degree * --cp-degree ({tp} * {cp} = {tp * cp})."
            )
        if args.seed is None:
            args.seed = 42  # all ranks must start from identical latents
    else:
        rank, dev = 0, torch.device("cuda")

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    hf_repo = HF_REPOS[args.model]
    log(f"Downloading pipeline from {hf_repo}")
    pipeline_path = pathlib.Path(snapshot_download(repo_id=hf_repo))
    log(f"Loading pipeline from {pipeline_path} …")

    if distributed:
        # Load on CPU first (a TP-sharded model may not fit one GPU), then place / shard.
        pipeline = Cosmos3OmniPipeline.from_pretrained(
            str(pipeline_path),
            torch_dtype=torch.bfloat16,
            enable_safety_checker=not args.disable_safety_checker,
        )
        qh = pipeline.transformer.config.num_attention_heads
        kv = pipeline.transformer.config.num_key_value_heads
        if kv % tp != 0:
            raise ValueError(f"--tp-degree ({tp}) must divide the {kv} KV heads.")
        if qh % world != 0:
            raise ValueError(f"--tp-degree * --cp-degree ({world}) must divide the {qh} query heads.")
        mesh = init_device_mesh("cuda", (tp, cp), mesh_dim_names=("tp", "cp"))
        if tp > 1:
            enable_cosmos3_tensor_parallel(pipeline.transformer, mesh["tp"])  # shard weights -> GPUs
        pipeline.to(dev)  # place the replicated remainder (embeddings, norms, VAE, ...)
        pipeline.transformer.set_attention_backend("native")  # GQA-capable backend
        if cp > 1:
            enable_cosmos3_context_parallel(pipeline.transformer, mesh["cp"])  # shard the sequence
        elif tp > 1:
            enable_cosmos3_flash_attention(pipeline.transformer)  # GQA-safe dense attention
        log(f"Parallelism: TP={tp} x CP={cp} over {world} GPUs.")
    else:
        pipeline = Cosmos3OmniPipeline.from_pretrained(
            str(pipeline_path),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            enable_safety_checker=not args.disable_safety_checker,
        )
    log("Pipeline loaded successfully.")

    if args.flow_shift is not None:
        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
        )

    output_dir = pathlib.Path(args.output)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed) if args.seed is not None else None

    if args.action_mode is not None:
        if args.vision_path is None:
            raise ValueError("--vision-path must point to a conditioning video for action modes.")
        if args.action_chunk_size is None:
            raise ValueError("--action-chunk-size is required for action modes.")
        video = load_video(args.vision_path)
        raw_actions = _load_action(args.action_path) if args.action_mode == "forward_dynamics" else None
        result = pipeline(
            prompt=args.prompt,
            action=CosmosActionCondition(
                mode=args.action_mode,
                chunk_size=args.action_chunk_size,
                domain_name=args.domain_name,
                resolution_tier=args.resolution_tier,
                raw_actions=raw_actions,
                video=video,
                view_point=args.view_point,
            ),
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            use_system_prompt=False,
            add_resolution_template=args.add_resolution_template,
            add_duration_template=args.add_duration_template,
            enable_safety_check=not args.no_safety_check,
        )
    elif args.video_path is not None:
        video = load_video(args.video_path)
        condition_frame_indexes_vision = (
            [int(i) for i in args.condition_frame_indexes_vision.split(",") if i.strip()]
            if args.condition_frame_indexes_vision is not None
            else [0, 1]
        )
        result = pipeline(
            prompt=args.prompt,
            video=video,
            condition_frame_indexes_vision=condition_frame_indexes_vision,
            condition_video_keep=args.condition_video_keep,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            enable_sound=args.enable_sound,
            guidance_scale=args.guidance_scale,
            generator=generator,
            add_resolution_template=args.add_resolution_template,
            add_duration_template=args.add_duration_template,
            enable_safety_check=not args.no_safety_check,
        )
    else:
        image = load_image(args.vision_path) if args.vision_path is not None else None
        result = pipeline(
            prompt=args.prompt,
            image=image,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            enable_sound=args.enable_sound,
            guidance_scale=args.guidance_scale,
            generator=generator,
            add_resolution_template=args.add_resolution_template,
            add_duration_template=args.add_duration_template,
            enable_safety_check=not args.no_safety_check,
        )

    # Every rank produces the same output under parallelism; only rank 0 writes it.
    if rank == 0:
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

        if result.action is not None:
            for action in result.action:
                action_path = output_dir / "sample_action.json"
                with open(action_path, "w") as f:
                    json.dump(action.tolist(), f)
                print(f"Saved: {action_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
