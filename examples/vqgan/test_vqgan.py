#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import json
import logging
import os
import shutil
import sys
import tempfile

import torch

from diffusers import VQModel
from diffusers.utils.testing_utils import require_timm


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


@require_timm
class TextToImage(ExamplesTestsAccelerate):
    @property
    def test_vqmodel_config(self):
        return {
            "_class_name": "VQModel",
            "_diffusers_version": "0.17.0.dev0",
            "act_fn": "silu",
            "block_out_channels": [
                32,
            ],
            "down_block_types": [
                "DownEncoderBlock2D",
            ],
            "in_channels": 3,
            "latent_channels": 4,
            "layers_per_block": 2,
            "norm_num_groups": 32,
            "norm_type": "spatial",
            "num_vq_embeddings": 32,
            "out_channels": 3,
            "sample_size": 32,
            "scaling_factor": 0.18215,
            "up_block_types": [
                "UpDecoderBlock2D",
            ],
            "vq_embed_dim": 4,
        }

    @property
    def test_discriminator_config(self):
        return {
            "_class_name": "Discriminator",
            "_diffusers_version": "0.27.0.dev0",
            "in_channels": 3,
            "cond_channels": 0,
            "hidden_channels": 8,
            "depth": 4,
        }

    def get_vq_and_discriminator_configs(self, tmpdir):
        vqmodel_config_path = os.path.join(tmpdir, "vqmodel.json")
        discriminator_config_path = os.path.join(tmpdir, "discriminator.json")
        with open(vqmodel_config_path, "w") as fp:
            json.dump(self.test_vqmodel_config, fp)
        with open(discriminator_config_path, "w") as fp:
            json.dump(self.test_discriminator_config, fp)
        return vqmodel_config_path, discriminator_config_path

    def test_vqmodel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vqmodel_config_path, discriminator_config_path = self.get_vq_and_discriminator_configs(tmpdir)
            test_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(
                os.path.isfile(os.path.join(tmpdir, "discriminator", "diffusion_pytorch_model.safetensors"))
            )
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "vqmodel", "diffusion_pytorch_model.safetensors")))

    def test_vqmodel_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vqmodel_config_path, discriminator_config_path = self.get_vq_and_discriminator_configs(tmpdir)
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            model = VQModel.from_pretrained(tmpdir, subfolder="checkpoint-2/vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4"},
            )

            # Run training script for 2 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --checkpointing_steps=1
                --resume_from_checkpoint={os.path.join(tmpdir, 'checkpoint-4')}
                --output_dir {tmpdir}
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # no checkpoint-2 -> check old checkpoints do not exist
            # check new checkpoints exist
            # In the current script, checkpointing_steps 1 is equivalent to checkpointing_steps 2 as after the generator gets trained for one step,
            # the discriminator gets trained and loss and saving happens after that. Thus we do not expect to get a checkpoint-5
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_vqmodel_checkpointing_use_ema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vqmodel_config_path, discriminator_config_path = self.get_vq_and_discriminator_configs(tmpdir)
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # check can run an intermediate checkpoint
            model = VQModel.from_pretrained(tmpdir, subfolder="checkpoint-2/vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 2 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --checkpointing_steps=1
                --resume_from_checkpoint={os.path.join(tmpdir, 'checkpoint-4')}
                --output_dir {tmpdir}
                --use_ema
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # no checkpoint-2 -> check old checkpoints do not exist
            # check new checkpoints exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_vqmodel_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vqmodel_config_path, discriminator_config_path = self.get_vq_and_discriminator_configs(tmpdir)
            # Run training script with checkpointing
            # max_train_steps == 6, checkpointing_steps == 2, checkpoints_total_limit == 2
            # Should create checkpoints at steps 2, 4, 6
            # with checkpoint at step 2 deleted

            initial_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # check checkpoint directories exist
            # checkpoint-2 should have been deleted
            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-4", "checkpoint-6"})

    def test_vqmodel_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vqmodel_config_path, discriminator_config_path = self.get_vq_and_discriminator_configs(tmpdir)
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --checkpointing_steps=2
                --output_dir {tmpdir}
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            # resume and we should try to checkpoint at 6, where we'll have to remove
            # checkpoint-2 and checkpoint-4 instead of just a single previous checkpoint

            resume_run_args = f"""
                examples/vqgan/train_vqgan.py
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 32
                --image_column image
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 8
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --model_config_name_or_path {vqmodel_config_path}
                --discriminator_config_name_or_path {discriminator_config_path}
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint={os.path.join(tmpdir, 'checkpoint-4')}
                --checkpoints_total_limit=2
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            model = VQModel.from_pretrained(tmpdir, subfolder="vqmodel")
            image = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
            _ = model(image)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-6", "checkpoint-8"},
            )
