# Copyright 2025 HuggingFace Inc.
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

"""Unit tests for StableAudio3EulerScheduler."""

import tempfile
import unittest

import torch

from diffusers import StableAudio3EulerScheduler
from diffusers.schedulers.scheduling_stable_audio3_euler import StableAudio3EulerSchedulerOutput

from ..testing_utils import require_torch


def _scheduler(n=100, logsnr_min=-6.2, logsnr_max=2.0):
    s = StableAudio3EulerScheduler(num_inference_steps=n, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
    s.set_timesteps()
    return s


@require_torch
class TestStableAudio3EulerScheduler(unittest.TestCase):
    # ------------------------------------------------------------------
    def test_set_timesteps_shapes(self):
        s = _scheduler(n=100)
        self.assertEqual(s.sigmas.shape, (101,))  # N+1
        self.assertEqual(s.timesteps.shape, (100,))  # N

    # ------------------------------------------------------------------
    def test_sigmas_strictly_decreasing(self):
        s = _scheduler()
        diffs = s.sigmas[:-1] - s.sigmas[1:]
        self.assertTrue((diffs > 0).all(), "sigmas must be strictly decreasing")

    # ------------------------------------------------------------------
    def test_sigmas_match_logsnr_conversion(self):
        """Endpoints pinned to 1.0 / 0.0; interior points follow the logSNR->sigma conversion.

        This must match the reference `build_schedule` + `LogSNRShift(rate=0)`.
        """
        logsnr_min, logsnr_max = -6.2, 2.0
        n = 100
        s = _scheduler(n=n, logsnr_min=logsnr_min, logsnr_max=logsnr_max)
        self.assertAlmostEqual(s.sigmas[0].item(), 1.0, places=6)
        self.assertAlmostEqual(s.sigmas[-1].item(), 0.0, places=6)
        expected = torch.sigmoid(-torch.linspace(logsnr_min, logsnr_max, n + 1))
        self.assertTrue(torch.allclose(s.sigmas[1:-1], expected[1:-1], atol=1e-6))

    # ------------------------------------------------------------------
    def test_timesteps_are_sigmas_prefix(self):
        s = _scheduler()
        self.assertTrue(torch.allclose(s.timesteps, s.sigmas[:-1]))

    # ------------------------------------------------------------------
    def test_step_output_shape(self):
        s = _scheduler()
        B, C, T = 2, 4, 16
        sample = torch.randn(B, C, T)
        v = torch.randn(B, C, T)
        out = s.step(v, s.timesteps[0], sample)
        self.assertIsInstance(out, StableAudio3EulerSchedulerOutput)
        self.assertEqual(out.prev_sample.shape, (B, C, T))
        self.assertEqual(out.pred_original_sample.shape, (B, C, T))

    # ------------------------------------------------------------------
    def test_step_return_dict_false(self):
        s = _scheduler()
        sample = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        out = s.step(v, s.timesteps[0], sample, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)

    # ------------------------------------------------------------------
    def test_step_euler_update_formula(self):
        """prev_sample = x_t + (σ_next − σ_curr) · v  should hold exactly."""
        s = _scheduler()
        sample = torch.ones(1, 1, 4)
        v = torch.full((1, 1, 4), 0.5)
        t_curr = s.sigmas[0]
        t_next = s.sigmas[1]
        out = s.step(v, s.timesteps[0], sample, return_dict=False)
        expected = sample + (t_next - t_curr) * v
        self.assertTrue(torch.allclose(out[0], expected))

    # ------------------------------------------------------------------
    def test_step_pred_original_sample_formula(self):
        """x̂₀ = x_t − t·v  should hold exactly."""
        s = _scheduler()
        t = s.sigmas[0]
        sample = torch.ones(1, 1, 4)
        v = torch.full((1, 1, 4), 0.5)
        out = s.step(v, s.timesteps[0], sample, return_dict=False)
        expected_x0 = sample - t * v
        self.assertTrue(torch.allclose(out[1], expected_x0))

    # ------------------------------------------------------------------
    def test_step_is_deterministic(self):
        """Euler sampling has no stochasticity: the generator argument must not change the output."""
        sample = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        s1 = _scheduler()
        out1 = s1.step(v, s1.timesteps[0], sample, generator=torch.Generator().manual_seed(1)).prev_sample
        s2 = _scheduler()
        out2 = s2.step(v, s2.timesteps[0], sample, generator=torch.Generator().manual_seed(2)).prev_sample
        self.assertTrue(torch.allclose(out1, out2))

    # ------------------------------------------------------------------
    def test_zero_velocity_is_noop(self):
        """v=0 means the flow ODE holds the sample fixed across every step."""
        s = _scheduler(n=20)
        x = torch.randn(1, 4, 8)
        x0 = x.clone()
        for t in s.timesteps:
            x = s.step(torch.zeros_like(x), t, x).prev_sample
        self.assertTrue(torch.allclose(x, x0))

    # ------------------------------------------------------------------
    def test_full_loop_no_nan(self):
        s = _scheduler(n=20)
        x = torch.randn(1, 4, 16)
        for t in s.timesteps:
            v = torch.randn_like(x)
            x = s.step(v, t, x).prev_sample
        self.assertFalse(x.isnan().any(), "NaN detected in output")
        self.assertFalse(x.isinf().any(), "Inf detected in output")

    # ------------------------------------------------------------------
    def test_set_timesteps_override(self):
        s = _scheduler(n=100)
        s.set_timesteps(10)
        self.assertEqual(s.num_inference_steps, 10)
        self.assertEqual(s.sigmas.shape, (11,))
        self.assertEqual(s.timesteps.shape, (10,))

    # ------------------------------------------------------------------
    def test_step_without_set_timesteps_raises(self):
        s = StableAudio3EulerScheduler()
        with self.assertRaises(RuntimeError):
            s.step(torch.zeros(1, 1, 4), 0.5, torch.zeros(1, 1, 4))

    # ------------------------------------------------------------------
    def test_add_noise_linear_interpolation(self):
        """add_noise must implement (1-t)*x0 + t*eps exactly."""
        s = _scheduler()
        x0 = torch.ones(2, 4, 8)
        eps = torch.zeros(2, 4, 8)
        t = torch.tensor([0.5, 0.0])
        noisy = s.add_noise(x0, eps, t)
        self.assertAlmostEqual(noisy[0].mean().item(), 0.5, places=5)
        self.assertAlmostEqual(noisy[1].mean().item(), 1.0, places=5)

    # ------------------------------------------------------------------
    def test_scale_model_input_is_identity(self):
        s = _scheduler()
        x = torch.randn(2, 4, 8)
        self.assertTrue(torch.equal(s.scale_model_input(x), x))

    # ------------------------------------------------------------------
    def test_len(self):
        s = StableAudio3EulerScheduler(num_inference_steps=100)
        self.assertEqual(len(s), 100)

    # ------------------------------------------------------------------
    def test_config_roundtrip(self):
        s = StableAudio3EulerScheduler(num_inference_steps=50, logsnr_min=-5.0, logsnr_max=1.5)
        with tempfile.TemporaryDirectory() as d:
            s.save_pretrained(d)
            s2 = StableAudio3EulerScheduler.from_pretrained(d)
        self.assertEqual(s2.config.num_inference_steps, 50)
        self.assertEqual(s2.config.logsnr_min, -5.0)
        self.assertEqual(s2.config.logsnr_max, 1.5)

    # ------------------------------------------------------------------
    def test_device_placement(self):
        s = _scheduler()
        s.set_timesteps(device="cpu")
        self.assertEqual(s.sigmas.device.type, "cpu")
        self.assertEqual(s.timesteps.device.type, "cpu")

    # ------------------------------------------------------------------
    def test_step_index_advances(self):
        s = _scheduler()
        sample = torch.randn(2, 4, 8)
        v = torch.randn(2, 4, 8)
        self.assertEqual(s._step_index, 0)
        s.step(v, s.timesteps[0], sample)
        self.assertEqual(s._step_index, 1)
        s.step(v, s.timesteps[1], sample)
        self.assertEqual(s._step_index, 2)

    # ------------------------------------------------------------------
    def test_set_timesteps_resets_step_index(self):
        s = _scheduler()
        sample = torch.randn(1, 4, 8)
        v = torch.randn(1, 4, 8)
        s.step(v, s.timesteps[0], sample)
        s.set_timesteps()  # reset
        self.assertEqual(s._step_index, 0)


if __name__ == "__main__":
    unittest.main()
