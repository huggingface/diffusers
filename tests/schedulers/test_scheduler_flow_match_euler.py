import unittest

import torch

from diffusers import FlowMatchEulerDiscreteScheduler

from ..testing_utils import torch_device


class FlowMatchEulerDiscreteSchedulerTest(unittest.TestCase):
    def test_index_for_timestep_handles_float_precision(self):
        # `timesteps` is derived from `sigmas * num_train_timesteps` in float32, so conceptually integer
        # timesteps can carry a small rounding error (e.g. `254.00001`). `index_for_timestep` must still
        # resolve them instead of raising an `IndexError` from an empty exact-equality lookup. See
        # https://github.com/huggingface/diffusers/issues/11749.
        scheduler = FlowMatchEulerDiscreteScheduler()
        timesteps = scheduler.timesteps.to(torch_device)

        # locate an entry that is not bit-exact to its rounded integer value
        rounded = timesteps.round()
        mismatched = (timesteps != rounded).nonzero()
        self.assertGreater(mismatched.numel(), 0, "expected at least one timestep with float rounding error")
        index = int(mismatched[0])
        integer_timestep = rounded[index]

        self.assertFalse(torch.equal(timesteps[index], integer_timestep))
        self.assertEqual(scheduler.index_for_timestep(integer_timestep, timesteps), index)

    def test_index_for_timestep_matches_exact_value(self):
        # an exact element of the schedule must still resolve to its own index (no drift to a neighbor).
        scheduler = FlowMatchEulerDiscreteScheduler()
        timesteps = scheduler.timesteps.to(torch_device)

        for index in (0, len(timesteps) // 2, len(timesteps) - 1):
            self.assertEqual(scheduler.index_for_timestep(timesteps[index], timesteps), index)

    def test_index_for_timestep_rejects_unknown_timestep(self):
        # a value that is not in the schedule (beyond float tolerance) must not be silently matched.
        scheduler = FlowMatchEulerDiscreteScheduler()
        timesteps = scheduler.timesteps.to(torch_device)

        unknown = (timesteps[0] + timesteps[1]) / 2  # halfway between two steps
        with self.assertRaises(IndexError):
            scheduler.index_for_timestep(unknown, timesteps)

    def test_scale_noise_with_integer_timesteps(self):
        # the training path of `scale_noise` (`begin_index is None`) is commonly called with integer
        # timesteps; it must not raise from the float-precision lookup. See issue #11749.
        scheduler = FlowMatchEulerDiscreteScheduler()
        self.assertIsNone(scheduler.begin_index)

        sample = torch.randn(2, 4, 8, 8).to(torch_device)
        noise = torch.randn_like(sample)
        timesteps = torch.tensor([254, 500], device=torch_device)

        scaled = scheduler.scale_noise(sample, timesteps, noise)
        self.assertEqual(scaled.shape, sample.shape)
        self.assertTrue(torch.isfinite(scaled).all())
