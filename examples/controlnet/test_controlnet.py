# coding=utf-8
# Copyright 2024 HuggingFace Inc.
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


class ControlNet(ExamplesTestsAccelerate):
    def test_controlnet_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_controlnet_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            --max_train_steps=6
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4", "checkpoint-6"},
            )

            resume_run_args = f"""
            examples/controlnet/train_controlnet.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet
            --max_train_steps=8
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-6
            --checkpoints_total_limit=2
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-6", "checkpoint-8"})


class ControlNetSDXL(ExamplesTestsAccelerate):
    def test_controlnet_sdxl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/controlnet/train_controlnet_sdxl.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-xl-pipe
            --dataset_name=hf-internal-testing/fill10
            --output_dir={tmpdir}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --controlnet_model_name_or_path=hf-internal-testing/tiny-controlnet-sdxl
            --max_train_steps=4
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "diffusion_pytorch_model.safetensors")))
