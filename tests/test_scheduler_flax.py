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
from typing import Dict, List, Tuple

import jax.numpy as jnp
from jax import random

from diffusers import (
    FlaxDDIMScheduler,
    FlaxDDPMScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
    FlaxScoreSdeVeScheduler,
)


class FlaxSchedulerCommonTest(unittest.TestCase):
    scheduler_classes = ()
    forward_default_kwargs = ()

    @property
    def dummy_sample(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        key1, key2 = random.split(random.PRNGKey(0))
        sample = random.uniform(key1, (batch_size, num_channels, height, width))

        return sample, key2

    @property
    def dummy_sample_deter(self):
        batch_size = 4
        num_channels = 3
        height = 8
        width = 8

        num_elems = batch_size * num_channels * height * width
        sample = jnp.arange(num_elems)
        sample = sample.reshape(num_channels, height, width, batch_size)
        sample = sample / num_elems
        return jnp.transpose(sample, (3, 0, 1, 2))

    def get_scheduler_config(self):
        raise NotImplementedError

    def dummy_model(self):
        def model(sample, t, *args):
            return sample * t / (t + 1)

        return model

    def check_over_configs(self, time_step=0, **config):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config(**config)
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.state

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)
                new_state = new_scheduler.state

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def check_over_forward(self, time_step=0, **forward_kwargs):
        kwargs = dict(self.forward_default_kwargs)
        kwargs.update(forward_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.state

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)
                new_state = new_scheduler.state

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, time_step, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, time_step, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_from_pretrained_save_pretrained(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            sample, key = self.dummy_sample
            residual = 0.1 * sample

            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.state

            with tempfile.TemporaryDirectory() as tmpdirname:
                scheduler.save_config(tmpdirname)
                new_scheduler = scheduler_class.from_config(tmpdirname)
                new_state = new_scheduler.state

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
                new_scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output = scheduler.step(state, residual, 1, sample, key, **kwargs).prev_sample
            new_output = new_scheduler.step(new_state, residual, 1, sample, key, **kwargs).prev_sample

            assert jnp.sum(jnp.abs(output - new_output)) < 1e-5, "Scheduler outputs are not identical"

    def test_step_shape(self):
        kwargs = dict(self.forward_default_kwargs)

        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.state

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            output_0 = scheduler.step(state, residual, 0, sample, key, **kwargs).prev_sample
            output_1 = scheduler.step(state, residual, 1, sample, key, **kwargs).prev_sample

            self.assertEqual(output_0.shape, sample.shape)
            self.assertEqual(output_0.shape, output_1.shape)

    def test_scheduler_outputs_equivalence(self):
        def set_nan_tensor_to_zero(t):
            return t.at[t != t].set(0)

        def recursive_check(tuple_object, dict_object):
            if isinstance(tuple_object, (List, Tuple)):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object, dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif isinstance(tuple_object, Dict):
                for tuple_iterable_value, dict_iterable_value in zip(tuple_object.values(), dict_object.values()):
                    recursive_check(tuple_iterable_value, dict_iterable_value)
            elif tuple_object is None:
                return
            else:
                self.assertTrue(
                    jnp.allclose(set_nan_tensor_to_zero(tuple_object), set_nan_tensor_to_zero(dict_object), atol=1e-5),
                    msg=(
                        "Tuple and dict output are not equal. Difference:"
                        f" {jnp.max(jnp.abs(tuple_object - dict_object))}. Tuple has `nan`:"
                        f" {jnp.isnan(tuple_object).any()} and `inf`: {jnp.isinf(tuple_object)}. Dict has"
                        f" `nan`: {jnp.isnan(dict_object).any()} and `inf`: {jnp.isinf(dict_object)}."
                    ),
                )

        kwargs = dict(self.forward_default_kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", None)

        for scheduler_class in self.scheduler_classes:
            scheduler_config = self.get_scheduler_config()
            scheduler = scheduler_class(**scheduler_config)
            state = scheduler.state

            sample, key = self.dummy_sample
            residual = 0.1 * sample

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_dict = scheduler.step(state, residual, 0, sample, key, **kwargs)

            if num_inference_steps is not None and hasattr(scheduler, "set_timesteps"):
                scheduler.set_timesteps(num_inference_steps)
            elif num_inference_steps is not None and not hasattr(scheduler, "set_timesteps"):
                kwargs["num_inference_steps"] = num_inference_steps

            outputs_tuple = scheduler.step(state, residual, 0, sample, key, return_dict=False, **kwargs)

            recursive_check(outputs_tuple[0], outputs_dict.prev_sample)


class FlaxDDPMSchedulerTest(FlaxSchedulerCommonTest):
    scheduler_classes = (FlaxDDPMScheduler,)

    def get_scheduler_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "beta_schedule": "linear",
            "variance_type": "fixed_small",
            "clip_sample": True,
        }

        config.update(**kwargs)
        return config

    def test_timesteps(self):
        for timesteps in [1, 5, 100, 1000]:
            self.check_over_configs(num_train_timesteps=timesteps)

    def test_betas(self):
        for beta_start, beta_end in zip([0.0001, 0.001, 0.01, 0.1], [0.002, 0.02, 0.2, 2]):
            self.check_over_configs(beta_start=beta_start, beta_end=beta_end)

    def test_schedules(self):
        for schedule in ["linear", "squaredcos_cap_v2"]:
            self.check_over_configs(beta_schedule=schedule)

    def test_variance_type(self):
        for variance in ["fixed_small", "fixed_large", "other"]:
            self.check_over_configs(variance_type=variance)

    def test_clip_sample(self):
        for clip_sample in [True, False]:
            self.check_over_configs(clip_sample=clip_sample)

    def test_time_indices(self):
        for t in [0, 500, 999]:
            self.check_over_forward(time_step=t)

    def test_variance(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)

        assert jnp.sum(jnp.abs(scheduler._get_variance(0) - 0.0)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(487) - 0.00979)) < 1e-5
        assert jnp.sum(jnp.abs(scheduler._get_variance(999) - 0.02)) < 1e-5

    def test_full_loop_no_noise(self):
        scheduler_class = self.scheduler_classes[0]
        scheduler_config = self.get_scheduler_config()
        scheduler = scheduler_class(**scheduler_config)
        state = scheduler.state

        num_trained_timesteps = len(scheduler)

        model = self.dummy_model()
        sample = self.dummy_sample_deter
        key1, key2 = random.split(random.PRNGKey(0))

        for t in reversed(range(num_trained_timesteps)):
            # 1. predict noise residual
            residual = model(sample, t)

            # 2. predict previous mean of sample x_t-1
            output = scheduler.step(state, residual, t, sample, key1)
            pred_prev_sample = output.prev_sample
            state = output.state
            key1, key2 = random.split(key2)

            # if t > 0:
            #     noise = self.dummy_sample_deter
            #     variance = scheduler.get_variance(t) ** (0.5) * noise
            #
            # sample = pred_prev_sample + variance
            sample = pred_prev_sample

        result_sum = jnp.sum(jnp.abs(sample))
        result_mean = jnp.mean(jnp.abs(sample))

        assert abs(result_sum - 255.1113) < 1e-2
        assert abs(result_mean - 0.332176) < 1e-3
