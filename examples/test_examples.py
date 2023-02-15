# coding=utf-8
# Copyright 2022 HuggingFace Inc..
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
import subprocess
import sys
import tempfile
import unittest
from typing import List

from accelerate.utils import write_basic_config

from diffusers import DiffusionPipeline, UNet2DConditionModel


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


# These utils relate to ensuring the right error message is received when running scripts
class SubprocessCallException(Exception):
    pass


def run_command(command: List[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class ExamplesTestsAccelerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.mkdtemp()
        cls.configPath = os.path.join(cls._tmpdir, "default_config.yml")

        write_basic_config(save_location=cls.configPath)
        cls._launch_args = ["accelerate", "launch", "--config_file", cls.configPath]

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(cls._tmpdir)

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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_textual_inversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --train_data_dir docs/source/en/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token a
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "learned_embeds.bin")))

    def test_dreambooth(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --instance_data_dir docs/source/en/imgs
                --instance_prompt photo
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))

    def test_dreambooth_checkpointing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instance_prompt = "photo"
            pretrained_model_name_or_path = "hf-internal-testing/tiny-stable-diffusion-pipe"

            # Run training script with checkpointing
            # max_train_steps == 5, checkpointing_steps == 2
            # Should create checkpoints at steps 2, 4

            initial_run_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 5
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
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # check checkpoint directories exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))

            # check can run an intermediate checkpoint
            unet = UNet2DConditionModel.from_pretrained(tmpdir, subfolder="checkpoint-2/unet")
            pipe = DiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # Remove checkpoint 2 so that we can check only later checkpoints exist after resuming
            shutil.rmtree(os.path.join(tmpdir, "checkpoint-2"))

            # Run training script for 7 total steps resuming from checkpoint 4

            resume_run_args = f"""
                examples/dreambooth/train_dreambooth.py
                --pretrained_model_name_or_path {pretrained_model_name_or_path}
                --instance_data_dir docs/source/en/imgs
                --instance_prompt {instance_prompt}
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 1
                --max_train_steps 7
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
            pipe = DiffusionPipeline.from_pretrained(tmpdir, safety_checker=None)
            pipe(instance_prompt, num_inference_steps=2)

            # check old checkpoints do not exist
            self.assertFalse(os.path.isdir(os.path.join(tmpdir, "checkpoint-2")))

            # check new checkpoints exist
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-4")))
            self.assertTrue(os.path.isdir(os.path.join(tmpdir, "checkpoint-6")))

    def test_text_to_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/text_to_image/train_text_to_image.py
                --pretrained_model_name_or_path hf-internal-testing/tiny-stable-diffusion-pipe
                --dataset_name hf-internal-testing/dummy_image_text_data
                --resolution 64
                --center_crop
                --random_flip
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
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
