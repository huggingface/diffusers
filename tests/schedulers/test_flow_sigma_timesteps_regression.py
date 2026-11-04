
import pytest
import torch

from diffusers import (
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
)


@pytest.mark.parametrize(
    "scheduler_class",
    [UniPCMultistepScheduler, DPMSolverMultistepScheduler],
)
def test_flow_sigma_timesteps_match_sigmas(scheduler_class):
    scheduler = scheduler_class(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=3.0,
    )

    scheduler.set_timesteps(8)

    expected = (
        scheduler.sigmas[:-1] * scheduler.config.num_train_timesteps
    )

    # Flow-matching timesteps must retain fractional values.
    assert scheduler.timesteps.is_floating_point()

    # The model's conditioning timesteps must match the
    # sigma values used by the numerical solver.
    torch.testing.assert_close(
        scheduler.timesteps.cpu(),
        expected.cpu(),
        rtol=1e-5,
        atol=1e-4,
    )
