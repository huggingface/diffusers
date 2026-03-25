#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video, load_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageDataset(Dataset):
    """Dataset that loads images and their corresponding text prompts.

    Expects a directory with:
        <filename>.jpg / .jpeg / .png  — the conditioning image
        <filename>.txt                 — the prompt text
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = []

        for filename in sorted(os.listdir(data_dir)):
            stem, ext = os.path.splitext(filename)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue
            img_path = os.path.join(data_dir, filename)
            txt_path = os.path.join(data_dir, stem + ".txt")
            if not os.path.exists(txt_path):
                print(f"WARNING: no prompt file found for {img_path}, skipping.")
                continue
            self.samples.append((img_path, txt_path, stem))

        if len(self.samples) == 0:
            raise ValueError(f"No valid image/prompt pairs found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, txt_path, stem = self.samples[idx]
        image = load_image(img_path)
        with open(txt_path) as f:
            prompt = f.read().strip()
        return {
            "image": image,
            "prompt": prompt,
            "stem": stem,
        }


def collate_fn(batch):
    """Keep images as a list (PIL images can't be stacked into a tensor)."""
    return {
        "images": [item["image"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "stems": [item["stem"] for item in batch],
    }


def arch_invariant_rand(shape, dtype, device, seed=None):
    rng = np.random.RandomState(seed)
    random_array = rng.standard_normal(shape).astype(np.float32)
    return torch.from_numpy(random_array).to(dtype=dtype, device=device)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval Cosmos Predict 2.5 with optional LoRA weights.")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory with image/prompt pairs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated outputs.")
    parser.add_argument(
        "--model_id", type=str, default="nvidia/Cosmos-Predict2.5-2B", help="HuggingFace model repository."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="diffusers/base/post-trained",
        choices=["diffusers/base/post-trained", "diffusers/base/pre-trained"],
    )
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA weights directory.")
    parser.add_argument("--num_output_frames", type=int, default=93, help="1 for image output, 93 for video output.")
    parser.add_argument("--num_steps", type=int, default=36, help="Number of inference steps.")
    parser.add_argument("--height", type=int, default=704, help="Output height in pixels (must be divisible by 16).")
    parser.add_argument("--width", type=int, default=1280, help="Output width in pixels (must be divisible by 16).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of samples per batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker processes.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt. Defaults to the pipeline's built-in negative prompt.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = ImageDataset(args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    print(f"Found {len(dataset)} examples.")

    class MockSafetyChecker:
        def to(self, *args, **kwargs):
            return self

        def check_text_safety(self, *args, **kwargs):
            return True

        def check_video_safety(self, video):
            return video

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        safety_checker=MockSafetyChecker(),
    )

    if args.lora_dir is not None:
        pipe.load_lora_weights(args.lora_dir)
        pipe.fuse_lora(lora_scale=1.0)
        print(f"Loaded LoRA weights from {args.lora_dir}")

    latent_shape = pipe.get_latent_shape_cthw(args.height, args.width, args.num_output_frames)
    noises = arch_invariant_rand(
        (args.batch_size, *latent_shape), dtype=torch.float32, device=args.device, seed=args.seed
    )
    progress = tqdm(total=len(dataset), desc="Generating")
    for batch in dataloader:
        images = batch["images"]
        prompts = batch["prompts"]
        stems = batch["stems"]

        for image, prompt, stem in zip(images, prompts, stems):
            frames = pipe(
                image=image,
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                num_frames=args.num_output_frames,
                num_inference_steps=args.num_steps,
                height=args.height,
                width=args.width,
                latents=noises,
            ).frames[0]  # NOTE: batch_size == 1

            out_path = os.path.join(args.output_dir, f"{stem}.mp4")
            export_to_video(frames, out_path, fps=16)

            tqdm.write(f"  Saved to: {out_path}")
            progress.update(1)


if __name__ == "__main__":
    main()
