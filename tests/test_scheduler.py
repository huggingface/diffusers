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


import tempfile
import unittest

import numpy as np
import torch

from diffusers import DDIMScheduler, DDPMScheduler


torch.backends.cuda.matmul.allow_tf32 = False


class SchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def dummy_image(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        image = np.random.rand(batch_size, num_channels, height, width)

        return image

    @property
    def dummy_image_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        image = np.arange(num_elems)
        image = image.reshape(num_channels, height, width, batch_size)
        image = image / num_elems
        image = image.transpose(3, 0, 1, 2)

        return image

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(image, t, *args):
            return image * t / (t + 1)

        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            scheduler_class = self.scheduler_classes[0]
            image = self.dummy_image
            residual = 0.1 * image

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)

            output = scheduler.step(residual, image, time_step, **kwargs)
            new_output = new_scheduler.step(residual, image, time_step, **kwargs)

            assert np.sum(np.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        for scheduler_class in self.scheduler_classes:
            scheduler_class = self.scheduler_classes[0]
            image = self.dummy_image
            residual = 0.1 * image

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)

            output = scheduler.step(residual, image, time_step, **kwargs)
            new_output = new_scheduler.step(residual, image, time_step, **kwargs)

            assert np.sum(np.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_pretrained_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            image = self.dummy_image
            residual = 0.1 * image

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)

            output = scheduler.step(residual, image, 1, **kwargs)
            new_output = new_scheduler.step(residual, image, 1, **kwargs)

            assert np.sum(np.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            image = self.dummy_image
            residual = 0.1 * image

            output_0 = scheduler.step(residual, image, 0, **kwargs)
            output_1 = scheduler.step(residual, image, 1, **kwargs)

            self.assertEqual(output_0.shape, image.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_pytorch_equal_numpy(self):
        kwargs = dict(self.forward_default_kwargs)

        for scheduler_class in self.scheduler_classes:
            image = self.dummy_image
            residual = 0.1 * image

            image_pt = torch.tensor(image)
            residual_pt = 0.1 * image_pt

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)

            scheduler_pt = scheduler_class(tensor_format="pt", **scheduler_config)

            output = scheduler.step(residual, image, 1, **kwargs)
            output_pt = scheduler_pt.step(residual_pt, image_pt, 1, **kwargs)

            assert np.sum(np.abs(output - output_pt.numpy())) < 1e-5, "Scheduler outputs are not identical"


class DDPMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_predicted_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_image(self):
        for clip_predicted_sample in [True, False]:
            self.check_over_configs(clip_predicted_sample=clip_predicted_sample)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert np.sum(np.abs(scheduler.get_variance(0) - 0.0)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(487) - 0.00979)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        image = self.dummy_image_deter

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(image, t)

            # 2. predict previous mean of image x_t-1
            pred_prev_image = scheduler.step(residual, image, t)

            if t > 0:
                noise = self.dummy_image_deter
                variance = scheduler.get_variance(t) ** (0.5) * noise

            image = pred_prev_image + variance

        result_sum = np.sum(np.abs(image))
        result_mean = np.mean(np.abs(image))

        assert result_sum.item() - 732.9947 < 1e-3
        assert result_mean.item() - 0.9544 < 1e-3


class DDIMSchedulerTest(SchedulerCommonTest):
    scheduler_classes = (DDIMScheduler,)
    forward_default_kwargs = (("num_inference_steps", 50), ("eta", 0.0))

    def get_scheduler_config(self, **kwargs):
        config = {
            "timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "clip_predicted_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_clip_image(self):
        for clip_predicted_sample in [True, False]:
            self.check_over_configs(clip_predicted_sample=clip_predicted_sample)

    def test_time_indices(self):
        for t in [1, 10, 49]:
            self.check_over_forward(time_step=t)

    def test_inference_steps(self):
        for t, num_inference_steps in zip([1, 10, 50], [10, 50, 500]):
            self.check_over_forward(time_step=t, num_inference_steps=num_inference_steps)

    def test_eta(self):
        for t, eta in zip([1, 10, 49], [0.0, 0.5, 1.0]):
            self.check_over_forward(time_step=t, eta=eta)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert np.sum(np.abs(scheduler.get_variance(0, 50) - 0.0)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(21, 50) - 0.14771)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(49, 50) - 0.32460)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(0, 1000) - 0.0)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(487, 1000) - 0.00979)) < 1e-5
        assert np.sum(np.abs(scheduler.get_variance(999, 1000) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        num_inference_steps, eta = 10, 0.1
        num_trained_timesteps = len(scheduler)

        inference_step_times = range(0, num_trained_timesteps, num_trained_timesteps // num_inference_steps)

        model = self.dummy_model()
        image = self.dummy_image_deter

        for t in reversed(range(num_inference_steps)):
            residual = model(image, inference_step_times[t])

            pred_prev_image = scheduler.step(residual, image, t, num_inference_steps, eta)

            variance = 0
            if eta > 0:
                noise = self.dummy_image_deter
                variance = scheduler.get_variance(t, num_inference_steps) ** (0.5) * eta * noise

            image = pred_prev_image + variance

        result_sum = np.sum(np.abs(image))
        result_mean = np.mean(np.abs(image))

        assert result_sum.item() - 270.6214 < 1e-3
        assert result_mean.item() - 0.3524 < 1e-3
