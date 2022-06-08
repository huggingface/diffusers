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


import random
import tempfile
import unittest
import os
from distutils.util import strtobool

import torch

from diffusers import GaussianDDPMScheduler, UNetModel
from diffusers.pipeline_utils import DiffusionPipeline
from models.vision.ddpm.modeling_ddpm import DDPM
from models.vision.ddim.modeling_ddim import DDIM


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

    return torch.tensor(data=values, dtype=torch.float).view(shape).contiguous()


class ModelTesterMixin(unittest.TestCase):
    @property
    def dummy_input(self):
        batch_size = 4
        num_channels = 3
        sizes = (32, 32)

        noise = floats_tensor((batch_size, num_channels) + sizes)
        time_step = torch.tensor([10])

        return (noise, time_step)

    def test_from_pretrained_save_pretrained(self):
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_pretrained(tmpdirname)
            new_model = UNetModel.from_pretrained(tmpdirname)

        dummy_input = self.dummy_input

        image = model(*dummy_input)
        new_image = new_model(*dummy_input)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    def test_from_pretrained_hub(self):
        model = UNetModel.from_pretrained("fusing/ddpm_dummy")

        image = model(*self.dummy_input)

        assert image is not None, "Make sure output is not None"


class SamplerTesterMixin(unittest.TestCase):

    @slow
    def test_sample(self):
        generator = torch.manual_seed(0)

        # 1. Load models
        scheduler = GaussianDDPMScheduler.from_config("fusing/ddpm-lsun-church")
        model = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

        # 2. Sample gaussian noise
        image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=torch_device, generator=generator)

        # 3. Denoise
        for t in reversed(range(len(scheduler))):
            # i) define coefficients for time step t
            clipped_image_coeff = 1 / torch.sqrt(scheduler.get_alpha_prod(t))
            clipped_noise_coeff = torch.sqrt(1 / scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - scheduler.get_alpha_prod(t - 1)) * torch.sqrt(scheduler.get_alpha(t)) / (1 - scheduler.get_alpha_prod(t))
            clipped_coeff = torch.sqrt(scheduler.get_alpha_prod(t - 1)) * scheduler.get_beta(t) / (1 - scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            with torch.no_grad():
                noise_residual = model(image, t)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clipped_image_coeff * image - clipped_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clipped_coeff * pred_mean + image_coeff * image

            # iv) sample variance
            prev_variance = scheduler.sample_variance(t, prev_image.shape, device=torch_device, generator=generator)

            # v) sample  x_{t-1} ~ N(prev_image, prev_variance)
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        # Note: The better test is to simply check with the following lines of code that the image is sensible
        # import PIL
        # import numpy as np
        # image_processed = image.cpu().permute(0, 2, 3, 1)
        # image_processed = (image_processed + 1.0) * 127.5
        # image_processed = image_processed.numpy().astype(np.uint8)
        # image_pil = PIL.Image.fromarray(image_processed[0])
        # image_pil.save("test.png")

        assert image.shape == (1, 3, 256, 256)
        image_slice = image[0, -1, -3:, -3:].cpu()
        expected_slice = torch.tensor([-0.1636, -0.1765, -0.1968, -0.1338, -0.1432, -0.1622, -0.1793, -0.2001, -0.2280])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    def test_sample_fast(self):
        # 1. Load models
        generator = torch.manual_seed(0)

        scheduler = GaussianDDPMScheduler.from_config("fusing/ddpm-lsun-church", timesteps=10)
        model = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

        # 2. Sample gaussian noise
        image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=torch_device, generator=generator)

        # 3. Denoise
        for t in reversed(range(len(scheduler))):
            # i) define coefficients for time step t
            clipped_image_coeff = 1 / torch.sqrt(scheduler.get_alpha_prod(t))
            clipped_noise_coeff = torch.sqrt(1 / scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - scheduler.get_alpha_prod(t - 1)) * torch.sqrt(scheduler.get_alpha(t)) / (1 - scheduler.get_alpha_prod(t))
            clipped_coeff = torch.sqrt(scheduler.get_alpha_prod(t - 1)) * scheduler.get_beta(t) / (1 - scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            with torch.no_grad():
                noise_residual = model(image, t)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clipped_image_coeff * image - clipped_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clipped_coeff * pred_mean + image_coeff * image

            # iv) sample variance
            prev_variance = scheduler.sample_variance(t, prev_image.shape, device=torch_device, generator=generator)

            # v) sample  x_{t-1} ~ N(prev_image, prev_variance)
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        assert image.shape == (1, 3, 256, 256)
        image_slice = image[0, -1, -3:, -3:].cpu()
        expected_slice = torch.tensor([-0.0304, -0.1895, -0.2436, -0.9837, -0.5422, 0.1931, -0.8175, 0.0862, -0.7783])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2


class PipelineTesterMixin(unittest.TestCase):

    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
        schedular = GaussianDDPMScheduler(timesteps=10)

        ddpm = DDPM(model, schedular)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPM.from_pretrained(tmpdirname)

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(0)
        new_image = new_ddpm(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_from_pretrained_hub(self):
        model_path = "fusing/ddpm-cifar10"

        ddpm = DDPM.from_pretrained(model_path)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path)

        ddpm.noise_scheduler.num_timesteps = 10
        ddpm_from_hub.noise_scheduler.num_timesteps = 10

        generator = torch.manual_seed(0)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(0)
        new_image = ddpm_from_hub(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"

    @slow
    def test_ddpm_cifar10(self):
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        ddpm = DDPM.from_pretrained(model_id)
        image = ddpm(generator=generator)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor([0.2250, 0.3375, 0.2360, 0.0930, 0.3440, 0.3156, 0.1937, 0.3585, 0.1761])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2

    @slow
    def test_ddim_cifar10(self):
        generator = torch.manual_seed(0)
        model_id = "fusing/ddpm-cifar10"

        ddim = DDIM.from_pretrained(model_id)
        image = ddim(generator=generator, eta=0.0)

        image_slice = image[0, -1, -3:, -3:].cpu()

        assert image.shape == (1, 3, 32, 32)
        expected_slice = torch.tensor([-0.7688, -0.7690, -0.7597, -0.7660, -0.7713, -0.7531, -0.7009, -0.7098, -0.7350])
        assert (image_slice.flatten() - expected_slice).abs().max() < 1e-2
