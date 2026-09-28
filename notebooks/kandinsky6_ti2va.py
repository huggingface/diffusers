#!/usr/bin/env python3
"""Kandinsky 6 text/image-to-video-and-audio generation.

Script version of kandinsky6_ti2va.ipynb, for running on a headless server.

Usage:
    python kandinsky6_ti2va.py --gpu-id 0
    python kandinsky6_ti2va.py --model k6pro --batch-size 1 --no-magcache
    python kandinsky6_ti2va.py --gen-mode i2va --image assets/i2va_input.png

Authenticate with the Hub before running if the model repo is gated:
    huggingface-cli login
  or:
    export HF_TOKEN=hf_...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffusers import Kandinsky6TI2VAPipeline
from diffusers.hooks import MagCacheConfig
from diffusers.models.transformers.transformer_kandinsky6 import (
    Kandinsky6FusedTransformerDecoderBlock,
    Kandinsky6TransformerDecoderBlock,
    Kandinsky6TransformerEncoderBlock,
)
from diffusers.utils import encode_video

MODELS = {
    "k6pro": "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers",
    "k6pro_distill": "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="k6pro_distill")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gen-mode", choices=["t2va", "i2va"], default="t2va")
    parser.add_argument("--image", type=Path, default=None, help="Reference image, required for --gen-mode i2va.")
    parser.add_argument(
        "--prompt",
        default=(
            "A news anchor delivers the evening news in a clear, standard, and neutral tone, saying: "
            "<S>Good evening, ladies and gentlemen, and welcome to the evening news.<E>"
        ),
    )
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality, noisy audio")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=864)
    parser.add_argument("--num-frames", type=int, default=121)
    parser.add_argument("--sample-fps", type=float, default=24.0)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1137)
    parser.add_argument("--expand-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--magcache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable MagCache. Defaults to on for non-distilled models, off for distilled ones.",
    )
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diffusers"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gen_mode == "i2va" and args.image is None:
        raise SystemExit("--gen-mode i2va requires --image <path>")

    model_path = MODELS[args.model]
    use_magcache = bool("distill" not in args.model) if args.magcache is None else args.magcache
    backend = "native" if args.use_compile else "_flash_3"

    pipe = Kandinsky6TI2VAPipeline.from_pretrained(model_path, dtype=torch.bfloat16)
    pipe.transformer.set_attention_backend(backend)

    # Compile after loading the weights and before CPU offload.
    if args.use_compile:
        torch.set_float32_matmul_precision("high")
        for block in pipe.transformer.modules():
            if isinstance(block, Kandinsky6FusedTransformerDecoderBlock):
                block.forward = torch.compile(block.forward)
            elif isinstance(block, (Kandinsky6TransformerEncoderBlock, Kandinsky6TransformerDecoderBlock)):
                block.forward = torch.compile(block.forward, mode="max-autotune-no-cudagraphs", dynamic=True)

    is_piflow = pipe.scheduler.__class__.__name__ == "PiflowScheduler"
    num_inference_steps = 16 if is_piflow else 50
    guidance_scale = 1.0 if is_piflow else 5.0
    print(f"Scheduler: {pipe.scheduler.__class__.__name__}; steps: {num_inference_steps}; guidance: {guidance_scale}")

    output_suffix = ""
    output_suffix += "_piflow" if is_piflow else ""
    output_suffix += "_expand" if args.expand_prompts else ""
    output_suffix += "_compile" if args.use_compile else ""
    output_suffix += "_magcache" if use_magcache else ""
    output_path = args.output_dir / args.model
    output_path.mkdir(parents=True, exist_ok=True)

    # Diffusers' MagCacheConfig, sourced from the metadata the checkpoint carries in its
    # own config. K6 stores conditional/unconditional coefficients interleaved, so the
    # configuration counts both CFG forwards.
    magcache_meta = pipe.transformer.config.get("magcache")
    if use_magcache and magcache_meta is not None:
        mag_ratios = magcache_meta["mag_ratios"]
        if isinstance(mag_ratios, dict):
            mag_ratios = mag_ratios[args.gen_mode]
        diffusers_magcache = MagCacheConfig(
            threshold=magcache_meta["threshold"],
            max_skip_steps=magcache_meta["max_skip_steps"],
            retention_ratio=magcache_meta["retention_ratio"],
            num_inference_steps=2 * num_inference_steps,
            mag_ratios=list(mag_ratios),
        )
        if pipe.transformer.is_cache_enabled:
            pipe.transformer.disable_cache()
        pipe.transformer.enable_cache(diffusers_magcache)
        print(f"MagCache enabled: {pipe.transformer.is_cache_enabled}")
        print(f"MagCache coefficients ({args.gen_mode}): {len(mag_ratios)}; threshold: {magcache_meta['threshold']}")
    else:
        use_magcache = False
        if getattr(pipe.transformer, "is_cache_enabled", False):
            pipe.transformer.disable_cache()

    # Model CPU offload keeps one active component on GPU at a time.
    pipe.enable_model_cpu_offload(gpu_id=args.gpu_id)

    print(f"Qwen context length: {args.max_sequence_length}")
    print(f"Audio sample rate: {pipe.audio_sample_rate}; video FPS: {args.sample_fps}")

    prompts = [args.prompt] * args.batch_size
    negative_prompts = [args.negative_prompt] * args.batch_size
    images = [str(args.image)] * args.batch_size if args.gen_mode == "i2va" else None

    result = pipe(
        prompt=prompts,
        image=images,
        negative_prompt=negative_prompts,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        num_inference_steps=num_inference_steps,
        max_sequence_length=args.max_sequence_length,
        guidance_scale=guidance_scale,
        sample_audio=True,
        expand_prompts=args.expand_prompts,
        generator=torch.Generator().manual_seed(args.seed),
        output_type="pt",
    )
    frames = result.frames
    audios = result.audio
    print(f"decoded frames: {tuple(frames.shape)}")
    print(f"audio streams: {len(audios)}, samples in first stream: {audios[0].shape[0]}")

    final_output_dir = output_path / f"bs{args.batch_size}"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    for i, (video, audio) in enumerate(zip(frames, audios)):
        batch_suffix = f"_batchid{i}" if args.batch_size > 1 else ""
        video_path = (
            final_output_dir / f"kandinsky6_{args.gen_mode}_{num_inference_steps}steps{output_suffix}{batch_suffix}.mp4"
        )
        video = video.permute(1, 2, 3, 0)
        audio_samples = torch.as_tensor(audio)[:, None].repeat(1, 2)
        encode_video(
            video,
            fps=int(args.sample_fps),
            output_path=str(video_path),
            audio=audio_samples,
            audio_sample_rate=int(pipe.audio_sample_rate),
        )
        print(f"saved: {video_path}")


if __name__ == "__main__":
    main()
