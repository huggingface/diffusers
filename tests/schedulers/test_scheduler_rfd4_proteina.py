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

import pytest
import torch

from diffusers import RFD4ProteinaScheduler


def scheduler(schedule_mode="log", noise_mode="1/t", stochastic=False):
    """A tiny backbone-style scheduler. With `stochastic=False`, `t_lim_ode_below=1` keeps every interval on the
    noise-free branch; `stochastic=True` uses the released thresholds."""
    config = {
        "schedule_mode": schedule_mode,
        "noise_mode": noise_mode,
        "tsr_k": 16.0,
        "tsr_sigma": 3.0,
        "tail_steps": 0,
        "t_lim_ode": 0.98,
        "t_lim_ode_below": 0.02 if stochastic else 1.0,
    }
    return RFD4ProteinaScheduler(**config)


def inputs():
    sample = torch.tensor([[[0.2, -0.4, 0.6], [2.0, 3.0, 4.0]]])
    mask = torch.tensor([[True, False]])
    return sample, mask, torch.full_like(sample, 0.25)


def test_schedule_and_config_round_trip(tmp_path):
    original = scheduler(stochastic=True)
    original.save_pretrained(tmp_path)
    loaded = RFD4ProteinaScheduler.from_pretrained(tmp_path, local_files_only=True)
    original.set_timesteps(2)
    loaded.set_timesteps(2)
    torch.testing.assert_close(loaded.timesteps, original.timesteps, rtol=0, atol=0)
    torch.testing.assert_close(loaded._noise, original._noise, rtol=0, atol=0)
    # Released schedules start near zero (below `t_lim_ode_below`) and end at 1.
    assert loaded.timesteps[0] < 0.02
    assert float(loaded._times[-1]) == 1.0
    assert loaded.step_index == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA scalar-placement coverage")
def test_cuda_schedule_keeps_coefficients_on_cpu():
    wrapper = scheduler(stochastic=True)
    wrapper.set_timesteps(2, device="cuda")

    assert wrapper.timesteps.device.type == "cuda"
    assert wrapper._times.device.type == "cpu"
    assert wrapper._noise.device.type == "cpu"


@pytest.mark.parametrize("schedule_mode, noise_mode", [("log", "1/t"), ("power", "tan")])
def test_partial_schedule_is_exact_nonlinear_full_grid_tail(schedule_mode, noise_mode):
    full = scheduler(schedule_mode, noise_mode, stochastic=True)
    full.set_timesteps(4)
    partial = scheduler(schedule_mode, noise_mode, stochastic=True)
    partial.set_timesteps(2, start_time=float(full._times[2]))
    torch.testing.assert_close(partial._times, full._times[2:], rtol=0, atol=0)
    torch.testing.assert_close(partial._noise, full._noise[2:], rtol=0, atol=0)


def test_first_step_matches_temporal_score_rescaling():
    wrapper = scheduler(stochastic=True)
    wrapper.set_timesteps(2)
    sample, mask, velocity = inputs()
    actual = wrapper.step(velocity, wrapper.timesteps[0], sample, mask, return_dict=False)[0]
    # The first time (1e-3) lies below `t_lim_ode_below`, so the interval is noise free: the velocity is converted to a
    # score, rescaled by the temporal score ratio, converted back and integrated with one Euler step.
    t = wrapper.timesteps[0]
    snr = t**2 / (1 - t) ** 2
    k, sigma = wrapper.config.tsr_k, wrapper.config.tsr_sigma
    ratio = (snr * sigma**2 + 1) / (snr * sigma**2 / k + 1)
    score = (t * velocity - sample) / (1 - t)
    rescaled = (sample + (1 - t) * score * ratio) / t
    expected = (sample + rescaled * (wrapper._times[1] - wrapper._times[0])) * mask[..., None]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert torch.equal(actual[:, 1], torch.zeros_like(actual[:, 1]))


def test_add_noise_interpolates_between_noise_and_data():
    wrapper = scheduler()
    clean = torch.ones(2, 3, 3)
    noise = torch.zeros_like(clean)
    torch.testing.assert_close(wrapper.add_noise(clean, noise, 0.25), torch.full_like(clean, 0.25))
    torch.testing.assert_close(
        wrapper.add_noise(clean, noise, torch.tensor([0.0, 0.5]))[:, 0, 0], torch.tensor([0.0, 0.5])
    )
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        wrapper.add_noise(clean, noise, 1.0)


def test_step_order_and_generator_validation():
    wrapper = scheduler()
    sample, mask, velocity = inputs()
    with pytest.raises(ValueError, match="set_timesteps"):
        wrapper.step(velocity, torch.zeros(()), sample, mask)
    wrapper.set_timesteps(2)
    with pytest.raises(ValueError, match="next element"):
        wrapper.step(velocity, wrapper.timesteps[1], sample, mask)
    assert wrapper.step_index == 0
    repeated = torch.Generator()
    with pytest.raises(ValueError, match="distinct"):
        wrapper.step(
            velocity.expand(2, -1, -1),
            wrapper.timesteps[0],
            sample.expand(2, -1, -1),
            mask.expand(2, -1),
            generator=[repeated, repeated],
        )


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
def test_stochastic_step_default_generator_matches_explicit_stream(device):
    explicit, implicit = scheduler(stochastic=True), scheduler(stochastic=True)
    explicit.set_timesteps(2, device=device)
    implicit.set_timesteps(2, device=device)
    sample, mask, velocity = (value.to(device) for value in inputs())
    generator = torch.Generator(device=device).manual_seed(71)
    initial_rng = generator.get_state().clone()
    devices = [torch.device(device)] if device.startswith("cuda") else []
    with torch.random.fork_rng(devices=devices):
        if devices:
            torch.cuda.set_rng_state(initial_rng, device)
        else:
            torch.set_rng_state(initial_rng)
        left, right = sample, sample
        for timestep in explicit.timesteps:
            left = explicit.step(velocity, timestep, left, mask, generator=generator).prev_sample
            right = implicit.step(velocity, timestep, right, mask).prev_sample
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        endpoint = torch.cuda.get_rng_state(device) if devices else torch.get_rng_state()
        assert torch.equal(endpoint, generator.get_state())
        assert not torch.equal(endpoint, initial_rng), "The test must exercise a stochastic interval."


@pytest.mark.parametrize("schedule_mode, noise_mode", [("uniform", "1/t"), ("log", "ode")])
def test_options_outside_the_released_recipe_are_rejected(schedule_mode, noise_mode):
    with pytest.raises(ValueError, match="must be"):
        scheduler(schedule_mode, noise_mode)
