# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from diffusers import AutoencoderRAE
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402
from train_rae_dit import (  # noqa: E402
    build_transforms,
    collate_fn,
    compute_resume_offsets,
    maybe_load_resumed_scheduler,
    should_skip_resumed_batch,
)


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


def _create_unique_class_dataset(dataset_dir: Path, resolution: int, num_samples: int):
    for sample_idx in range(num_samples):
        class_dir = dataset_dir / f"class_{sample_idx:02d}"
        class_dir.mkdir(parents=True, exist_ok=True)
        color = ((40 * sample_idx) % 256, (80 * sample_idx) % 256, (120 * sample_idx) % 256)
        image = Image.new("RGB", (resolution, resolution), color=color)
        image.save(class_dir / f"sample_{sample_idx}.png")


def _collect_class_label_trace(
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

    raise AssertionError(f"Expected to record {expected_microbatches} microbatches, but only collected {len(trace)}.")


class RAEDiT(ExamplesTestsAccelerate):
    def _create_tiny_rae(self, tmpdir):
        model = AutoencoderRAE(
            encoder_type="mae",
            encoder_hidden_size=64,
            encoder_patch_size=4,
            encoder_num_hidden_layers=1,
            encoder_input_size=16,
            patch_size=4,
            image_size=16,
            num_channels=3,
            decoder_hidden_size=64,
            decoder_num_hidden_layers=1,
            decoder_num_attention_heads=4,
            decoder_intermediate_size=128,
            encoder_norm_mean=[0.5, 0.5, 0.5],
            encoder_norm_std=[0.5, 0.5, 0.5],
            noise_tau=0.0,
            reshape_to_2d=True,
            scaling_factor=1.0,
        )
        output_dir = os.path.join(tmpdir, "tiny-rae")
        model.save_pretrained(output_dir, safe_serialization=False)
        return output_dir

    def _create_dataset(self, tmpdir, resolution=16):
        dataset_dir = os.path.join(tmpdir, "dataset")
        for class_idx in range(2):
            class_dir = os.path.join(dataset_dir, f"class_{class_idx:02d}")
            os.makedirs(class_dir, exist_ok=True)
            for image_idx in range(2):
                color = (
                    (50 * class_idx + 20 * image_idx) % 256,
                    (80 * class_idx + 30 * image_idx) % 256,
                    (110 * class_idx + 40 * image_idx) % 256,
                )
                image = Image.new("RGB", (resolution, resolution), color=color)
                image.save(os.path.join(class_dir, f"sample_{image_idx}.png"))
        return dataset_dir

    def test_train_rae_dit_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rae_dir = self._create_tiny_rae(tmpdir)
            dataset_dir = self._create_dataset(tmpdir)

            test_args = f"""
                examples/research_projects/rae_dit/train_rae_dit.py
                --pretrained_rae_model_name_or_path {rae_dir}
                --train_data_dir {dataset_dir}
                --output_dir {tmpdir}
                --resolution 16
                --center_crop
                --train_batch_size 1
                --dataloader_num_workers 0
                --max_train_steps 2
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_scheduler constant
                --lr_warmup_steps 0
                --encoder_hidden_size 32
                --decoder_hidden_size 32
                --encoder_num_layers 1
                --decoder_num_layers 1
                --encoder_num_attention_heads 4
                --decoder_num_attention_heads 4
                --mlp_ratio 2.0
                --num_train_timesteps 10
                """.split()

            run_command(self._launch_args + test_args)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "transformer", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "id2label.json")))

    def test_resume_batch_order_matches_uninterrupted_tail(self):
        seed = 123
        resolution = 16
        num_samples = 6
        train_batch_size = 1
        gradient_accumulation_steps = 2
        max_train_steps = 3
        resume_global_step = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "trace-dataset"
            _create_unique_class_dataset(dataset_dir, resolution=resolution, num_samples=num_samples)

            baseline_trace = _collect_class_label_trace(
                dataset_dir,
                seed=seed,
                resolution=resolution,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_train_steps=max_train_steps,
            )
            resumed_trace = _collect_class_label_trace(
                dataset_dir,
                seed=seed,
                resolution=resolution,
                train_batch_size=train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_train_steps=max_train_steps,
                resume_global_step=resume_global_step,
            )

        consumed_microbatches = resume_global_step * gradient_accumulation_steps
        self.assertEqual(resumed_trace, baseline_trace[consumed_microbatches:])

    def test_maybe_load_resumed_scheduler_prefers_checkpoint_config(self):
        args = SimpleNamespace(num_train_timesteps=999, flow_shift=2.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            scheduler_dir = os.path.join(tmpdir, "scheduler")
            FlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=7.0).save_pretrained(scheduler_dir)

            restored = maybe_load_resumed_scheduler(
                args=args,
                checkpoint_path=tmpdir,
                noise_scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=999, shift=2.5),
            )

        self.assertEqual(restored.config.num_train_timesteps, 10)
        self.assertEqual(restored.config.shift, 7.0)
        self.assertEqual(args.num_train_timesteps, 10)
        self.assertEqual(args.flow_shift, 7.0)
