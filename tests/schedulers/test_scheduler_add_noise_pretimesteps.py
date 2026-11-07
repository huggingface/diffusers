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
# See the License in the LICENSE file.

import unittest

import torch

from diffusers import (
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UniPCMultistepScheduler,
)


class SchedulerAddNoisePreSetTimestepsTest(unittest.TestCase):
    # Regression tests for https://github.com/huggingface/diffusers/issues/15021
    # add_noise() called *before* set_timesteps() must use the noise level of the
    # requested timestep, not of `999 - t`.
    def setUp(self):
        self.x0 = torch.ones(3, 4, 8, 8)
        self.noise = torch.randn(3, 4, 8, 8)
        self.timesteps = torch.tensor([200, 500, 800])

    def test_add_noise_before_set_timesteps_matches_ddpm(self):
        ddpm = DDPMScheduler(num_train_timesteps=1000)
        expected = ddpm.add_noise(self.x0, self.noise, self.timesteps)
        for sched_cls in (
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            UniPCMultistepScheduler,
            DEISMultistepScheduler,
        ):
            scheduler = sched_cls(num_train_timesteps=1000)
            noised = scheduler.add_noise(self.x0, self.noise, self.timesteps)
            self.assertEqual(noised.shape, self.x0.shape)
            max_err = (noised - expected).abs().max().item()
            self.assertLess(max_err, 1e-4, f"{sched_cls.__name__} VP mismatch (err={max_err})")

    def test_add_noise_flow_sigmas_before_set_timesteps(self):
        for sched_cls in (
            DPMSolverMultistepScheduler,
            DPMSolverSinglestepScheduler,
            UniPCMultistepScheduler,
            DEISMultistepScheduler,
        ):
            scheduler = sched_cls(num_train_timesteps=1000, use_flow_sigmas=True)
            noised = scheduler.add_noise(self.x0, self.noise, self.timesteps)
            frac = self.timesteps.to(noised.dtype) / 1000.0
            expected = (1 - frac).view(-1, 1, 1, 1) * self.x0 + frac.view(-1, 1, 1, 1) * self.noise
            max_err = (noised - expected).abs().max().item()
            self.assertLess(max_err, 1e-4, f"{sched_cls.__name__} flow mismatch (err={max_err})")

    def test_add_noise_after_set_timesteps_still_works(self):
        scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(10)
        noised = scheduler.add_noise(self.x0, self.noise, scheduler.timesteps[:3])
        self.assertEqual(noised.shape, self.x0.shape)
