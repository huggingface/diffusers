# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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
import os
import sys
import tempfile


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class Unconditional(ExamplesTestsAccelerate):
    def test_train_unconditional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 2
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_unconditional_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --checkpointing_steps=2
                --checkpoints_total_limit=2
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                # checkpoint-2 should have been deleted
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_unconditional_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            initial_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 1
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --checkpointing_steps=1
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-1", "checkpoint-2", "checkpoint-3", "checkpoint-4", "checkpoint-5", "checkpoint-6"},
            )

            resume_run_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name hf-internal-testing/dummy_image_class_data
                --model_config_name_or_path diffusers/ddpm_dummy
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 1
                --num_epochs 2
                --gradient_accumulation_steps 1
                --ddpm_num_inference_steps 2
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --resume_from_checkpoint=checkpoint-6
                --checkpointing_steps=2
                --checkpoints_total_limit=3
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check checkpoint directories exist
            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-8", "checkpoint-10", "checkpoint-12"},
            )
