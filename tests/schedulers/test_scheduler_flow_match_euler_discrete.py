import torch

from diffusers import FlowMatchEulerDiscreteScheduler


def test_stochastic_sampling_uses_s_noise_and_noise_clip_std():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0, stochastic_sampling=True)
    scheduler.set_timesteps(sigmas=[0.9, 0.5])

    sample = torch.ones((1, 1, 2, 2))
    model_output = torch.full_like(sample, 0.25)
    generator = torch.Generator(device="cpu").manual_seed(0)

    output = scheduler.step(
        model_output,
        scheduler.timesteps[0],
        sample,
        s_noise=2.0,
        noise_clip_std=0.5,
        generator=generator,
    ).prev_sample

    expected_noise = torch.randn(sample.shape, generator=torch.Generator(device="cpu").manual_seed(0))
    clip_value = 0.5 * expected_noise.std().item()
    expected_noise = expected_noise.clamp(min=-clip_value, max=clip_value)

    x0 = sample - scheduler.sigmas[0] * model_output
    expected = (1.0 - scheduler.sigmas[1]) * x0 + scheduler.sigmas[1] * expected_noise * 2.0

    torch.testing.assert_close(output, expected)
