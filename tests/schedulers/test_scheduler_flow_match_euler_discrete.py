import torch

from diffusers import FlowMatchEulerDiscreteScheduler


class TestFlowMatchEulerDiscreteSchedulerSigmaConsistency:
    """Regression tests for https://github.com/huggingface/diffusers/issues/13243"""

    def test_set_timesteps_no_double_shift(self):
        """Calling set_timesteps(num_train_timesteps) should reproduce the same sigmas as __init__.

        set_timesteps appends a terminal zero, so we compare only the first N values.
        """
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        sigmas_init = scheduler.sigmas.clone()

        scheduler.set_timesteps(1000)
        sigmas_after = scheduler.sigmas[:-1]  # drop appended terminal zero

        torch.testing.assert_close(sigmas_init, sigmas_after, atol=1e-6, rtol=1e-5)

    def test_set_timesteps_no_double_shift_various_shifts(self):
        """The fix holds for different shift values."""
        for shift in [1.0, 2.0, 3.0, 5.0]:
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
            sigmas_init = scheduler.sigmas.clone()

            scheduler.set_timesteps(1000)
            sigmas_after = scheduler.sigmas[:-1]

            torch.testing.assert_close(
                sigmas_init,
                sigmas_after,
                atol=1e-6,
                rtol=1e-5,
                msg=f"Sigma mismatch after set_timesteps with shift={shift}",
            )

    def test_set_timesteps_fewer_steps(self):
        """set_timesteps with fewer steps should produce sigmas within the original range."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
        scheduler.set_timesteps(50)

        # All sigmas should fall within [0, 1]
        assert scheduler.sigmas.min() >= 0.0
        assert scheduler.sigmas.max() <= 1.0 + 1e-6
