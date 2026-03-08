#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from train_rae_dit import build_transforms, collate_fn, compute_resume_offsets, should_skip_resumed_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Stage-2 RAE DiT mid-epoch resume batch ordering.")
    parser.add_argument("--seed", type=int, default=123, help="Seed used for the shuffled dataloader.")
    parser.add_argument("--resolution", type=int, default=16, help="Synthetic image resolution.")
    parser.add_argument("--num_samples", type=int, default=6, help="Number of unique samples/classes to create.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Microbatch size used by the trace harness.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps used to derive the mid-epoch checkpoint position.",
    )
    parser.add_argument("--max_train_steps", type=int, default=3, help="Total optimizer steps to trace.")
    parser.add_argument(
        "--resume_global_step",
        type=int,
        default=1,
        help="Optimizer step at which the synthetic run resumes from a checkpoint.",
    )
    return parser.parse_args()


def create_unique_class_dataset(dataset_dir: Path, resolution: int, num_samples: int):
    for sample_idx in range(num_samples):
        class_dir = dataset_dir / f"class_{sample_idx:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        color = (
            (40 * sample_idx) % 256,
            (80 * sample_idx) % 256,
            (120 * sample_idx) % 256,
        )
        image = Image.new("RGB", (resolution, resolution), color=color)
        image.save(class_dir / f"sample_{sample_idx}.png")


def collect_class_label_trace(
    dataset_dir: Path,
    *,
    seed: int,
    resolution: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    max_train_steps: int,
    resume_global_step: int = 0,
) -> list[int]:
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
    set_seed(seed)

    transform_args = SimpleNamespace(resolution=resolution, center_crop=True, random_flip=False)
    dataset = ImageFolder(dataset_dir, transform=build_transforms(transform_args))
    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    first_epoch = 0
    resume_step = 0
    should_resume = resume_global_step > 0

    if should_resume:
        first_epoch, resume_step = compute_resume_offsets(
            global_step=resume_global_step,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    expected_microbatches = (max_train_steps - resume_global_step) * gradient_accumulation_steps
    trace = []

    for epoch in range(first_epoch, num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if should_skip_resumed_batch(
                should_resume=should_resume,
                epoch=epoch,
                first_epoch=first_epoch,
                step=step,
                resume_step=resume_step,
            ):
                continue

            trace.extend(batch["class_labels"].tolist())
            if len(trace) >= expected_microbatches:
                return trace

    raise AssertionError(
        f"Expected to record {expected_microbatches} microbatches, but only collected {len(trace)}."
    )


def main():
    args = parse_args()

    if args.resume_global_step >= args.max_train_steps:
        raise ValueError(
            f"`resume_global_step` ({args.resume_global_step}) must be < `max_train_steps` ({args.max_train_steps})."
        )
    microbatches_per_epoch = args.num_samples // args.train_batch_size
    required_microbatches = args.max_train_steps * args.gradient_accumulation_steps
    if microbatches_per_epoch < required_microbatches:
        raise ValueError(
            "The verifier keeps the proof inside a single epoch. Increase `--num_samples` or decrease "
            "`--train_batch_size`, `--gradient_accumulation_steps`, or `--max_train_steps`."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = Path(tmpdir) / "trace-dataset"
        create_unique_class_dataset(dataset_dir, resolution=args.resolution, num_samples=args.num_samples)

        baseline_trace = collect_class_label_trace(
            dataset_dir,
            seed=args.seed,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_train_steps=args.max_train_steps,
        )
        resumed_trace = collect_class_label_trace(
            dataset_dir,
            seed=args.seed,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_train_steps=args.max_train_steps,
            resume_global_step=args.resume_global_step,
        )

        consumed_microbatches = args.resume_global_step * args.gradient_accumulation_steps
        expected_resumed_trace = baseline_trace[consumed_microbatches:]

        print(f"baseline_trace={json.dumps(baseline_trace)}")
        print(f"consumed_trace={json.dumps(baseline_trace[:consumed_microbatches])}")
        print(f"resumed_trace={json.dumps(resumed_trace)}")

        if resumed_trace != expected_resumed_trace:
            raise AssertionError(
                "Resumed batch order does not match the uninterrupted run tail. "
                f"Expected {expected_resumed_trace}, got {resumed_trace}."
            )

        print("resume batch order verified")


if __name__ == "__main__":
    main()
