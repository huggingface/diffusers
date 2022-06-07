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


global_rng = random.Random()
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


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
        generator = torch.Generator()
        generator = generator.manual_seed(6694729458485568)

        # 1. Load models
        scheduler = GaussianDDPMScheduler.from_config("fusing/ddpm-lsun-church")
        model = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

        # 2. Sample gaussian noise
        image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=torch_device, generator=generator)

        # 3. Denoise
        for t in reversed(range(len(scheduler))):
            # i) define coefficients for time step t
            clip_image_coeff = 1 / torch.sqrt(scheduler.get_alpha_prod(t))
            clip_noise_coeff = torch.sqrt(1 / scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - scheduler.get_alpha_prod(t - 1)) * torch.sqrt(scheduler.get_alpha(t)) / (1 - scheduler.get_alpha_prod(t))
            clip_coeff = torch.sqrt(scheduler.get_alpha_prod(t - 1)) * scheduler.get_beta(t) / (1 - scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            with torch.no_grad():
                noise_residual = model(image, t)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clip_image_coeff * image - clip_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clip_coeff * pred_mean + image_coeff * image

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
        assert (image_slice - torch.tensor([[-0.0598, -0.0611, -0.0506], [-0.0726, 0.0220, 0.0103], [-0.0723, -0.1310, -0.2458]])).abs().sum() < 1e-3

    def test_sample_fast(self):
        # 1. Load models
        generator = torch.Generator()
        generator = generator.manual_seed(6694729458485568)

        scheduler = GaussianDDPMScheduler.from_config("fusing/ddpm-lsun-church", timesteps=10)
        model = UNetModel.from_pretrained("fusing/ddpm-lsun-church").to(torch_device)

        # 2. Sample gaussian noise
        torch.manual_seed(0)
        image = scheduler.sample_noise((1, model.in_channels, model.resolution, model.resolution), device=torch_device, generator=generator)

        # 3. Denoise
        for t in reversed(range(len(scheduler))):
            # i) define coefficients for time step t
            clip_image_coeff = 1 / torch.sqrt(scheduler.get_alpha_prod(t))
            clip_noise_coeff = torch.sqrt(1 / scheduler.get_alpha_prod(t) - 1)
            image_coeff = (1 - scheduler.get_alpha_prod(t - 1)) * torch.sqrt(scheduler.get_alpha(t)) / (1 - scheduler.get_alpha_prod(t))
            clip_coeff = torch.sqrt(scheduler.get_alpha_prod(t - 1)) * scheduler.get_beta(t) / (1 - scheduler.get_alpha_prod(t))

            # ii) predict noise residual
            with torch.no_grad():
                noise_residual = model(image, t)

            # iii) compute predicted image from residual
            # See 2nd formula at https://github.com/hojonathanho/diffusion/issues/5#issue-896554416 for comparison
            pred_mean = clip_image_coeff * image - clip_noise_coeff * noise_residual
            pred_mean = torch.clamp(pred_mean, -1, 1)
            prev_image = clip_coeff * pred_mean + image_coeff * image

            # iv) sample variance
            prev_variance = scheduler.sample_variance(t, prev_image.shape, device=torch_device, generator=generator)

            # v) sample  x_{t-1} ~ N(prev_image, prev_variance)
            sampled_prev_image = prev_image + prev_variance
            image = sampled_prev_image

        assert image.shape == (1, 3, 256, 256)
        image_slice = image[0, -1, -3:, -3:].cpu()
        assert (image_slice - torch.tensor([[0.1746, 0.5125, -0.7920], [-0.5734, -0.2910, -0.1984], [0.4090, -0.7740, -0.3941]])).abs().sum() < 1e-3


class PipelineTesterMixin(unittest.TestCase):
    def test_from_pretrained_save_pretrained(self):
        # 1. Load models
        model = UNetModel(ch=32, ch_mult=(1, 2), num_res_blocks=2, attn_resolutions=(16,), resolution=32)
        schedular = GaussianDDPMScheduler(timesteps=10)

        ddpm = DDPM(model, schedular)

        with tempfile.TemporaryDirectory() as tmpdirname:
            ddpm.save_pretrained(tmpdirname)
            new_ddpm = DDPM.from_pretrained(tmpdirname)
        
        generator = torch.Generator()
        generator = generator.manual_seed(669472945848556)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(669472945848556)
        new_image = new_ddpm(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"
    

    @slow
    def test_from_pretrained_hub(self):
        model_path = "fusing/ddpm-cifar10"

        ddpm = DDPM.from_pretrained(model_path)
        ddpm_from_hub = DiffusionPipeline.from_pretrained(model_path)

        ddpm.noise_scheduler.num_timesteps = 10
        ddpm_from_hub.noise_scheduler.num_timesteps = 10


        generator = torch.Generator(device=torch_device)
        generator = generator.manual_seed(669472945848556)

        image = ddpm(generator=generator)
        generator = generator.manual_seed(669472945848556)
        new_image = ddpm_from_hub(generator=generator)

        assert (image - new_image).abs().sum() < 1e-5, "Models don't give the same forward pass"
