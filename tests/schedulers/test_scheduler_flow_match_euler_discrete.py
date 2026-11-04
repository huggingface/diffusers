# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 1.0,
        }
        config.update(**kwargs)
        return config

    def get_scheduler(self, sigmas=(1.0, 0.5), **kwargs):
        scheduler = self.scheduler_class(**self.get_default_config(**kwargs))
        scheduler.set_timesteps(sigmas=list(sigmas))
        return scheduler

    def test_per_token_uniform_schedule_multi_step_integration(self):
        scheduler = self.get_scheduler()
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(2, 3, 4, generator=generator)
        velocity = torch.randn(2, 3, 4, generator=generator)

        prev_sample = sample
        for timestep in scheduler.timesteps:
            per_token_timesteps = torch.full(sample.shape[:2], float(timestep))
            prev_sample = scheduler.step(
                velocity,
                timestep,
                prev_sample,
                per_token_timesteps=per_token_timesteps,
                return_dict=False,
            )[0]

        torch.testing.assert_close(prev_sample, sample - velocity)

    def test_per_token_heterogeneous_sigmas_and_terminal_token(self):
        scheduler = self.get_scheduler()
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(1, 3, 4, generator=generator)
        velocity = torch.randn(1, 3, 4, generator=generator)
        per_token_timesteps = torch.tensor([[1000.0, 500.0, 0.0]])

        prev_sample = scheduler.step(
            velocity,
            scheduler.timesteps[0],
            sample,
            per_token_timesteps=per_token_timesteps,
            return_dict=False,
        )[0]

        expected = sample.clone()
        expected[:, 0] = sample[:, 0] + (0.5 - 1.0) * velocity[:, 0]
        expected[:, 1] = sample[:, 1] + (0.0 - 0.5) * velocity[:, 1]
        expected[:, 2] = sample[:, 2]

        torch.testing.assert_close(prev_sample, expected)
        torch.testing.assert_close(prev_sample[:, 2], sample[:, 2])

    def test_per_token_stochastic_uses_same_velocity_sign(self):
        scheduler = self.get_scheduler(stochastic_sampling=True)
        inputs_generator = torch.Generator().manual_seed(0)
        sample = torch.randn(1, 3, 4, generator=inputs_generator)
        velocity = torch.randn(1, 3, 4, generator=inputs_generator)
        per_token_timesteps = torch.tensor([[1000.0, 500.0, 0.0]])
        seed = 1234

        generator = torch.Generator(device="cpu").manual_seed(seed)
        prev_sample = scheduler.step(
            velocity,
            scheduler.timesteps[0],
            sample,
            per_token_timesteps=per_token_timesteps,
            generator=generator,
            return_dict=False,
        )[0]

        noise = torch.randn(
            sample.shape,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            dtype=torch.float32,
        )

        current_sigma = torch.tensor([[[1.0], [0.5], [0.0]]])
        next_sigma = torch.tensor([[[0.5], [0.0], [0.0]]])
        x0 = sample - current_sigma * velocity
        expected = (1.0 - next_sigma) * x0 + next_sigma * noise

        torch.testing.assert_close(prev_sample, expected)
        torch.testing.assert_close(prev_sample[:, 2], sample[:, 2])
