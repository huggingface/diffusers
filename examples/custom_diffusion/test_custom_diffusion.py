# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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


class CustomDiffusion(ExamplesTestsAccelerate):
    def test_custom_diffusion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/custom_diffusion/train_custom_diffusion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt <new1>
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 1.0e-05
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --modifier_token <new1>
                --no_safe_serialization
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "pytorch_custom_diffusion_weights.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "<new1>.bin")))

    def test_custom_diffusion_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-4", "checkpoint-6"})

    def test_custom_diffusion_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=4
            --checkpointing_steps=2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            resume_run_args = f"""
            examples/custom_diffusion/train_custom_diffusion.py
            --pretrained_model_name_or_path=hf-internal-testing/tiny-stable-diffusion-pipe
            --instance_data_dir=docs/source/en/imgs
            --output_dir={tmpdir}
            --instance_prompt=<new1>
            --resolution=64
            --train_batch_size=1
            --modifier_token=<new1>
            --dataloader_num_workers=0
            --max_train_steps=8
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-4
            --checkpoints_total_limit=2
            --no_safe_serialization
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-6", "checkpoint-8"})
