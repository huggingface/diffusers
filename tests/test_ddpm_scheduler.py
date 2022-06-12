# coding=utf-8
# Copyright 2022 HuggingFace Inc.
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


import os
import random
import tempfile
import unittest
import numpy as np
from distutils.util import strtobool

import torch

from diffusers import GaussianDDPMScheduler, UNetModel, DDIMScheduler
from diffusers.configuration_utils import ConfigMixin
from diffusers.pipeline_utils import DiffusionPipeline
from models.vision.ddim.modeling_ddim import DDIM
from models.vision.ddpm.modeling_ddpm import DDPM
from models.vision.latent_diffusion.modeling_latent_diffusion import LatentDiffusion

global_rng = random.Random()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return np.random.randn(data=values, dtype=torch.float).view(shape).contiguous()


class SchedulerCommonTest(unittest.TestCase):

    scheduler_class = None

    @property
    def dummy_image(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        image = np.random.rand(batch_size, num_channels, height, width)

        return image

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(image, residual, t, *args):
            return (image + residual) * t / (t + 1)

        return model

    def test_from_pretrained_save_pretrained(self):
        scheduler_config = self.get_scheduler_config()
        scheduler = self.scheduler_class(scheduler_config())

        with tempfile.TemporaryDirectory() as tmpdirname:
            scheduler.save_pretrained(tmpdirname)
            new_scheduler = self.scheduler_class.from_config(tmpdirname)
