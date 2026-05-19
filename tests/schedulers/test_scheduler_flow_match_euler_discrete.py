import torch

from diffusers import FlowMatchEulerDiscreteScheduler


def test_set_timesteps_matches_init_with_static_shift():
    """Regression for #13243: with `use_dynamic_shifting=False` and matching
    `num_inference_steps`, `set_timesteps` must reproduce the same sigmas as
    `__init__`. Previously `sigma_min`/`sigma_max` stored the post-shift values,
    so `set_timesteps` rebuilt sigmas in shifted space and then applied the
    shift again, producing a different schedule for the same arguments.
    """
    num_train_timesteps = 1000
    shift = 3.0

    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=shift,
        use_dynamic_shifting=False,
    )
    init_sigmas = scheduler.sigmas.clone()

    scheduler.set_timesteps(num_inference_steps=num_train_timesteps)

    # set_timesteps appends a trailing 0.0 sentinel; compare the leading entries.
    assert torch.allclose(init_sigmas, scheduler.sigmas[:-1], atol=1e-5), (
        f"set_timesteps produced a different schedule than __init__ for the same "
        f"args. init[-3:]={init_sigmas[-3:].tolist()} "
        f"post[-3:]={scheduler.sigmas[-4:-1].tolist()}"
    )


def test_dynamic_shifting_is_unaffected():
    """With `use_dynamic_shifting=True` no static shift runs in `__init__`,
    so the pre-existing behavior must be preserved by the fix."""
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=True,
    )
    # Without dynamic shift inputs (mu) we just check sigma_min/max are the
    # untouched linear endpoints (1.0 at top, 1/N at bottom).
    assert scheduler.sigma_max == 1.0
    assert abs(scheduler.sigma_min - 1.0 / 1000) < 1e-6
