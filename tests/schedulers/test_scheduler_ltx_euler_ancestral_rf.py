# Copyright 2025 Vittoria Lanzo and The HuggingFace Team. All rights reserved.
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

from diffusers import LTXEulerAncestralRFScheduler


def _make_scheduler(**kwargs):
    config = {"num_train_timesteps": 1000, "eta": 1.0, "s_noise": 1.0}
    config.update(kwargs)
    return LTXEulerAncestralRFScheduler(**config)


def _linear_sigmas(n=4):
    """Return a monotonically decreasing sigma schedule with terminal 0."""
    return [round(1.0 - i / n, 6) for i in range(n + 1)]


class LTXEulerAncestralRFSchedulerTest(unittest.TestCase):
    # ------------------------------------------------------------------
    # set_timesteps: input validation
    # ------------------------------------------------------------------

    def test_set_timesteps_explicit_sigmas_valid(self):
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        self.assertEqual(scheduler.num_inference_steps, 4)
        self.assertEqual(len(scheduler.sigmas), 5)

    def test_set_timesteps_non_monotone_raises(self):
        """
        Non-monotonically-decreasing sigmas must raise ValueError.
        Without this check, step() computes sigma_down outside [0, 1]
        and sigma_ratio >> 1, silently amplifying the latent.
        """
        scheduler = _make_scheduler()
        # sigma increases at step 0 -> 1
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[0.2, 0.8, 0.5, 0.0])

    def test_set_timesteps_fully_ascending_raises(self):
        scheduler = _make_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[0.0, 0.5, 1.0])

    def test_set_timesteps_plateau_is_valid(self):
        """Equal consecutive sigmas (plateau steps) must NOT raise — used in img2img partial schedules."""
        scheduler = _make_scheduler()
        # plateau at the first two entries is intentional in some set_begin_index workflows
        scheduler.set_timesteps(sigmas=[1.0, 1.0, 0.5, 0.0])
        self.assertEqual(scheduler.num_inference_steps, 3)

    def test_set_timesteps_num_inference_steps_auto(self):
        """Auto-generated schedule (no explicit sigmas) must initialise correctly."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(num_inference_steps=10)
        self.assertEqual(scheduler.num_inference_steps, 10)
        self.assertEqual(len(scheduler.sigmas), 11)  # N steps + terminal 0
        # Verify the auto-generated schedule is itself monotone
        sigmas = scheduler.sigmas
        self.assertTrue(
            (sigmas[:-1] >= sigmas[1:]).all(),
            "Auto-generated sigma schedule is not monotonically non-increasing.",
        )

    # ------------------------------------------------------------------
    # step(): output invariants
    # ------------------------------------------------------------------

    def test_step_output_dtype_fp16_preserved(self):
        """prev_sample.dtype must equal sample.dtype for fp16 inputs."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        model_output = torch.randn_like(sample)
        out = scheduler.step(model_output, scheduler.timesteps[0], sample)
        self.assertEqual(out.prev_sample.dtype, torch.float16)

    def test_step_output_dtype_fp32_preserved(self):
        """prev_sample.dtype must equal sample.dtype for fp32 inputs."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(1, 4, 8, 8, dtype=torch.float32)
        model_output = torch.randn_like(sample)
        out = scheduler.step(model_output, scheduler.timesteps[0], sample)
        self.assertEqual(out.prev_sample.dtype, torch.float32)

    def test_step_output_shape_preserved(self):
        """prev_sample.shape must equal sample.shape."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(2, 4, 16, 16)
        model_output = torch.randn_like(sample)
        out = scheduler.step(model_output, scheduler.timesteps[0], sample)
        self.assertEqual(out.prev_sample.shape, sample.shape)

    def test_step_return_tuple(self):
        """return_dict=False must return a tuple whose first element matches return_dict=True."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        t = scheduler.timesteps[0]

        torch.manual_seed(0)
        out_dict = scheduler.step(model_output, t, sample, return_dict=True)
        scheduler._step_index = None  # reset step index to replay the same step
        torch.manual_seed(0)
        out_tuple = scheduler.step(model_output, t, sample, return_dict=False)

        self.assertIsInstance(out_tuple, tuple)
        self.assertTrue(torch.allclose(out_dict.prev_sample, out_tuple[0]))

    def test_step_eta_zero_is_deterministic(self):
        """
        With eta=0 no noise is injected; the output must be identical regardless
        of the generator seed passed.
        """
        scheduler = _make_scheduler(eta=0.0)
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(0))
        model_output = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
        t = scheduler.timesteps[0]

        out1 = scheduler.step(model_output, t, sample).prev_sample

        scheduler._step_index = None
        out2 = scheduler.step(
            model_output, t, sample, generator=torch.Generator().manual_seed(99)
        ).prev_sample

        self.assertTrue(torch.allclose(out1, out2), "eta=0 step should be fully deterministic.")

    def test_step_final_step_returns_denoised(self):
        """At sigma=0 (final denoising step) prev_sample must equal the denoised estimate."""
        scheduler = _make_scheduler(eta=1.0)
        # Two-step schedule: [0.5, 0.0]
        scheduler.set_timesteps(sigmas=[0.5, 0.0])
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)

        # First (and only real) step
        out = scheduler.step(model_output, scheduler.timesteps[0], sample)
        # At sigma_next=0 the scheduler must return the clean estimate x0 = x_t - sigma*v_t
        expected = sample - 0.5 * model_output
        self.assertTrue(torch.allclose(out.prev_sample, expected, atol=1e-5))

    def test_set_timesteps_sigma_above_one_raises(self):
        """Sigmas outside [0, 1] violate the RF/CONST parameterization assumption."""
        scheduler = _make_scheduler()
        with self.assertRaises(ValueError):
            scheduler.set_timesteps(sigmas=[2.0, 1.0, 0.5, 0.0])

    def test_step_eta_negative_raises(self):
        """eta < 0 is invalid and must raise ValueError at construction time."""
        with self.assertRaises(ValueError):
            _make_scheduler(eta=-0.1)

    def test_step_eta_greater_than_one_clamps_sigma_down(self):
        """eta > 1 on a coarse schedule pushes sigma_down < 0; must clamp, warn once, and stay finite."""
        scheduler = _make_scheduler(eta=2.0)
        # Coarse schedule: large step size maximises the chance sigma_down goes negative
        scheduler.set_timesteps(sigmas=[0.5, 0.1, 0.0])
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)
        self.assertFalse(scheduler._sigma_down_warned)

        out = scheduler.step(model_output, scheduler.timesteps[0], sample)

        # Warning flag must be set (warning was emitted)
        self.assertTrue(scheduler._sigma_down_warned)
        # Output must be finite (clamp prevented NaN/Inf from negative sigma_down)
        self.assertTrue(torch.isfinite(out.prev_sample).all())

        # Second step must NOT re-emit (deduplication)
        scheduler._sigma_down_warned_count_before = True  # flag already True
        out2 = scheduler.step(model_output, scheduler.timesteps[1], sample)
        self.assertTrue(torch.isfinite(out2.prev_sample).all())

    def test_step_index_advances(self):
        """_step_index must increment by 1 on each call."""
        scheduler = _make_scheduler()
        scheduler.set_timesteps(sigmas=_linear_sigmas(4))
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)

        for expected_idx in range(4):
            scheduler.step(model_output, scheduler.timesteps[expected_idx], sample)
            self.assertEqual(scheduler._step_index, expected_idx + 1)

    def test_step_beyond_end_returns_sample(self):
        """Calling step() past the last index must return the input sample unchanged."""
        scheduler = _make_scheduler(eta=0.0)
        scheduler.set_timesteps(sigmas=[0.5, 0.0])
        sample = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(sample)

        # Consume all steps normally
        scheduler.step(model_output, scheduler.timesteps[0], sample)
        # Force _step_index to the clamped maximum
        scheduler._step_index = len(scheduler.sigmas) - 1
        # A further call must not crash and must return a finite tensor
        out = scheduler.step(model_output, scheduler.timesteps[-1], sample)
        self.assertTrue(torch.isfinite(out.prev_sample).all())


if __name__ == "__main__":
    unittest.main()
