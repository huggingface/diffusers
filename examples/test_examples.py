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
from diffusers.utils import slow


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

    @slow
    def test_train_unconditional(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/unconditional_image_generation/train_unconditional.py
                --dataset_name huggan/few-shot-aurora
                --resolution 64
                --output_dir {tmpdir}
                --train_batch_size 4
                --num_epochs 1
                --gradient_accumulation_steps 1
                --learning_rate 1e-3
                --lr_warmup_steps 5
                --mixed_precision fp16
                """.split()

            run_command(self._launch_args + test_args, return_stdout=True)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "unet", "diffusion_pytorch_model.bin")))
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "scheduler", "scheduler_config.json")))
            # logging test
            self.assertTrue(len(os.listdir(os.path.join(tmpdir, "logs", "train_unconditional"))) > 0)

    @slow
    def test_textual_inversion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = f"""
                examples/textual_inversion/textual_inversion.py
                --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4
                --use_auth_token
                --train_data_dir docs/source/imgs
                --learnable_property object
                --placeholder_token <cat-toy>
                --initializer_token toy
                --resolution 64
                --train_batch_size 1
                --gradient_accumulation_steps 2
                --max_train_steps 10
                --learning_rate 5.0e-04
                --scale_lr
                --lr_scheduler constant
                --lr_warmup_steps 0
                --output_dir {tmpdir}
                --mixed_precision fp16
                """.split()

            run_command(self._launch_args + test_args)
            # save_pretrained smoke test
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "learned_embeds.bin")))
