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

import numpy as np
import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging

from ..testing_utils import CaptureLogger


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    """
    `set_timesteps` accepts `sigmas` and `timesteps` separately: `sigmas` drive the ODE and go through the
    scheduler's shifting, while explicitly passed `timesteps` are what the model is conditioned on and are kept as
    given. CogView4 and GLM-Image pass both (integer timesteps alongside the sigmas), so the two must be allowed to
    disagree.
    """

    scheduler_class = FlowMatchEulerDiscreteScheduler

    def get_default_config(self, **kwargs):
        config = {
            "num_train_timesteps": 1000,
            "shift": 3.0,
        }
        config.update(**kwargs)
        return config

    def test_explicit_timesteps_are_kept_when_sigmas_are_passed(self):
        # The CogView4-6B scheduler config at 1024x1024 (`mu` is what `calculate_shift` yields there).
        scheduler = self.scheduler_class(
            **self.get_default_config(
                shift=1.0,
                use_dynamic_shifting=True,
                base_shift=0.25,
                max_shift=0.75,
                base_image_seq_len=256,
                max_image_seq_len=4096,
                time_shift_type="linear",
            )
        )
        timesteps = np.linspace(1000, 1.0, 10).astype(np.int64).astype(np.float32)
        sigmas = timesteps / 1000

        scheduler.set_timesteps(10, sigmas=sigmas.tolist(), timesteps=timesteps.tolist(), mu=3.25)

        torch.testing.assert_close(scheduler.timesteps, torch.from_numpy(timesteps))
        # The sigmas are still shifted; only the timesteps are taken verbatim.
        self.assertFalse(torch.allclose(scheduler.sigmas[:-1], torch.from_numpy(sigmas)))
        self.assertAlmostEqual(scheduler.sigmas[-1].item(), 0.0)

    def test_sigmas_only_derive_the_timesteps_from_the_shifted_sigmas(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        sigmas = [1.0, 0.75, 0.5, 0.25, 0.001]

        scheduler.set_timesteps(sigmas=sigmas)

        torch.testing.assert_close(scheduler.timesteps, scheduler.sigmas[:-1] * 1000)
        self.assertFalse(torch.allclose(scheduler.sigmas[:-1], torch.tensor(sigmas)))

    def test_timesteps_only_are_kept_and_warn(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        timesteps = [1000.0, 750.0, 500.0, 250.0, 1.0]
        logger = logging.get_logger("diffusers.schedulers.scheduling_flow_match_euler_discrete")

        with CaptureLogger(logger) as cap_logger:
            scheduler.set_timesteps(timesteps=timesteps)

        torch.testing.assert_close(scheduler.timesteps, torch.tensor(timesteps))
        # The sigmas derived from the timesteps are shifted, so they no longer match the timesteps.
        self.assertFalse(torch.allclose(scheduler.sigmas[:-1] * 1000, torch.tensor(timesteps)))
        self.assertIn("`timesteps` were passed without `sigmas`", cap_logger.out)

    def test_no_warning_when_sigmas_are_passed(self):
        scheduler = self.scheduler_class(**self.get_default_config())
        logger = logging.get_logger("diffusers.schedulers.scheduling_flow_match_euler_discrete")

        with CaptureLogger(logger) as cap_logger:
            scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.001])
            scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.001], timesteps=[1000.0, 500.0, 1.0])

        self.assertEqual(cap_logger.out, "")
