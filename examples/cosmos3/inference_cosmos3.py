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
import json
import pathlib
import urllib.request

import torch
from huggingface_hub import snapshot_download

from diffusers import Cosmos3OmniPipeline, CosmosActionCondition, UniPCMultistepScheduler
from diffusers.utils import encode_video, export_to_video, load_image, load_video


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
    parser.add_argument("--raw-action-dim", type=int, default=None, help="Slice predicted action output to this size.")
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

    if args.flow_shift is not None:
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config, flow_shift=args.flow_shift)
        print(f"Scheduler flow_shift set to {args.flow_shift}.")

    output_dir = pathlib.Path(args.output)
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
                raw_action_dim=args.raw_action_dim,
                resolution_tier=args.resolution_tier,
                raw_actions=raw_actions,
                video=video,
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


if __name__ == "__main__":
    main()
