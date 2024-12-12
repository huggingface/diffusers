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
import shutil
import sys
import tempfile

from diffusers import DiffusionPipeline, SD3Transformer2DModel


sys.path.append("..")
from test_examples_utils import ExamplesTestsAccelerate, run_command  # noqa: E402


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class DreamBoothSD3(ExamplesTestsAccelerate):
    instance_data_dir = "docs/source/en/imgs"
    instance_prompt = "photo"
    pretrained_model_name_or_path = "hf-internal-testing/tiny-sd3-pipe"
    script_path = "examples/dreambooth/train_dreambooth_sd3.py"

    def test_dreambooth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 2
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "transformer", "diffusion_pytorch_model.safetensors")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_dreambooth_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run training script with checkpointing
            # max_train_steps == 4, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 4
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --seed=0
                """.split()

            run_command(self._launch_args + initial_run_args)

            # check can run the original fully trained output pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir)
            pipe(self.instance_prompt, num_inference_steps=1)

            # check checkpoint directories exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))

            # check can run an intermediate checkpoint
            transformer = SD3Transformer2DModel.from_pretrained(tmpdir, subfolder="checkpoint-2/transformer")
            pipe = DiffusionPipeline.from_pretrained(self.pretrained_model_name_or_path, transformer=transformer)
            pipe(self.instance_prompt, num_inference_steps=1)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                {self.script_path}
                --pretrained_model_name_or_path {self.pretrained_model_name_or_path}
                --instance_data_dir {self.instance_data_dir}
                --instance_prompt {self.instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 6
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --checkpointing_steps=2
                --resume_from_checkpoint=checkpoint-4
                --seed=0
                """.split()

            run_command(self._launch_args + resume_run_args)

            # check can run new fully trained pipeline
            pipe = DiffusionPipeline.from_pretrained(tmpdir)
            pipe(self.instance_prompt, num_inference_steps=1)

            # check old checkpoints do not exist
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))

            # check new checkpoints exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-6")))

    def test_dreambooth_checkpointing_checkpoints_total_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --instance_prompt={self.instance_prompt}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=6
            --checkpoints_total_limit=2
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-4", "checkpoint-6"},
            )

    def test_dreambooth_checkpointing_checkpoints_total_limit_removes_multiple_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --instance_prompt={self.instance_prompt}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=4
            --checkpointing_steps=2
            """.split()

            run_command(self._launch_args + test_args)

            self.assertEqual(
                {x for x in os.listdir(tmpdir) if "checkpoint" in x},
                {"checkpoint-2", "checkpoint-4"},
            )

            resume_run_args = f"""
            {self.script_path}
            --pretrained_model_name_or_path={self.pretrained_model_name_or_path}
            --instance_data_dir={self.instance_data_dir}
            --output_dir={tmpdir}
            --instance_prompt={self.instance_prompt}
            --resolution=64
            --train_batch_size=1
            --gradient_accumulation_steps=1
            --max_train_steps=8
            --checkpointing_steps=2
            --resume_from_checkpoint=checkpoint-4
            --checkpoints_total_limit=2
            """.split()

            run_command(self._launch_args + resume_run_args)

            self.assertEqual({x for x in os.listdir(tmpdir) if "checkpoint" in x}, {"checkpoint-6", "checkpoint-8"})
