# Copyright 2026 HuggingFace Inc.
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

from diffusers import SeaCacheConfig
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.hooks import HookRegistry, ModelHook
from diffusers.hooks.sea_cache import (
    _SEA_CACHE_BLOCK_HOOK,
    _SEA_CACHE_LEADER_BLOCK_HOOK,
    _SEA_CACHE_ROOT_HOOK,
    _apply_sea_filter,
)
from diffusers.models.cache_utils import CacheMixin


class CountingIdentity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states


class DummySeaBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.indicator_norm = CountingIdentity()
        self.calls = 0

    def forward(self, und_seq, gen_seq, rotary_emb=None):
        self.calls += 1
        return und_seq + 1, gen_seq * 2


class DummySeaTransformer(torch.nn.Module, CacheMixin):
    def __init__(self, num_layers=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([DummySeaBlock() for _ in range(num_layers)])

    def forward(self, hidden_states):
        shape = hidden_states.shape
        gen_seq = hidden_states.reshape(-1, shape[-1])
        und_seq = torch.zeros(1, shape[-1], device=hidden_states.device, dtype=hidden_states.dtype)
        for layer in self.layers:
            und_seq, gen_seq = layer(und_seq, gen_seq)
        return und_seq, gen_seq.reshape(shape)


@pytest.fixture(autouse=True)
def register_dummy_sea_block():
    TransformerBlockRegistry.register(
        DummySeaBlock,
        TransformerBlockMetadata(
            return_hidden_states_index=1,
            return_encoder_hidden_states_index=0,
            hidden_states_argument_name="gen_seq",
            encoder_hidden_states_argument_name="und_seq",
            hidden_states_norm_module_name="indicator_norm",
        ),
    )


def _metadata_callback(module, args, kwargs):
    hidden_states = kwargs.get("hidden_states", args[0] if args else None)
    if hidden_states is None:
        return None
    temporal, height, width = hidden_states.shape[:3]
    indexes = torch.arange(temporal * height * width, device=hidden_states.device)
    return [(indexes, (temporal, height, width))]


def _make_config(runtime, **kwargs):
    config_kwargs = {
        "threshold": 100.0,
        "retention_steps": 1,
        "cache_end_steps": 0,
        "indicator_source": "first_block",
        "current_step_callback": lambda: runtime["step"],
        "current_sigma_callback": lambda: runtime["sigma"],
        "num_inference_steps_callback": lambda: runtime["num_steps"],
        "metadata_callback": _metadata_callback,
    }
    config_kwargs.update(kwargs)
    return SeaCacheConfig(**config_kwargs)


def _get_root_hook(model):
    return model._diffusers_hook.get_hook(_SEA_CACHE_ROOT_HOOK)


@torch.no_grad()
def test_sea_cache_uses_independent_gates_and_histories_per_context():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 4}
    model = DummySeaTransformer()
    model.enable_cache(_make_config(runtime, threshold=0.5))
    root_hook = _get_root_hook(model)

    first_input = torch.ones(2, 2, 2, 1)
    with model.cache_context("cond"):
        _, first_output = model(first_input)
    torch.testing.assert_close(first_output, first_input * 8)
    assert [block.calls for block in model.layers] == [1, 1, 1]

    with model.cache_context("uncond"):
        _, first_uncond_output = model(first_input)
    torch.testing.assert_close(first_uncond_output, first_input * 8)
    assert [block.calls for block in model.layers] == [2, 2, 2]

    runtime.update(step=1)
    with model.cache_context("cond"):
        _, cached_output = model(first_input)
    torch.testing.assert_close(cached_output, first_input * 8)
    assert [block.calls for block in model.layers] == [2, 2, 2]

    changed_uncond_input = first_input * 100
    with model.cache_context("uncond"):
        _, uncond_output = model(changed_uncond_input)
    torch.testing.assert_close(uncond_output, changed_uncond_input * 8)
    assert [block.calls for block in model.layers] == [3, 3, 3]
    assert model.layers[0].indicator_norm.calls == 4
    assert root_hook.num_full_steps == 3
    assert root_hook.num_cached_steps == 1
    stats = model.get_cache_stats()
    assert stats["indicator_source"] == "first_block"
    assert stats["residual_order"] == 1
    assert stats["residual_boundary"] == "repeated_block_stack"
    assert stats["transformer_calls"] == 4
    assert stats["gate_evaluations"] == 4
    assert stats["gate_trace"] == [True, True, False, True]
    assert stats["branch_full_executions"] == {"cond": 1, "uncond": 2}
    assert stats["branch_reuses"] == {"cond": 1}
    assert stats["persistent_cache_bytes"] > 0

    model._reset_stateful_cache()
    assert root_hook.num_full_steps == 0
    assert root_hook.num_cached_steps == 0
    archived_stats = model.get_cache_stats()
    assert archived_stats["transformer_calls"] == 4
    assert archived_stats["gate_trace"] == [True, True, False, True]

    model.disable_cache()
    assert not model.is_cache_enabled


@torch.no_grad()
def test_sea_cache_max_consecutive_cached_forces_full_per_context():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 8}
    model = DummySeaTransformer()
    model.enable_cache(
        _make_config(
            runtime,
            threshold=100.0,
            retention_steps=0,
            cache_end_steps=0,
            max_consecutive_cached=2,
        )
    )

    for step in range(runtime["num_steps"]):
        runtime.update(step=step, sigma=0.9 - step * 0.1)
        with model.cache_context("cond"):
            model(torch.ones(1, 1, 1, 1))

    stats = model.get_cache_stats()
    assert stats["gate_trace"] == [True, False, False, True, False, False, True, False]
    assert stats["actual_full_executions"] == 3
    assert stats["actual_reuses"] == 5
    assert stats["max_consecutive_cached"] == 2
    assert stats["max_consecutive_cached_observed"] == 2
    assert stats["max_consecutive_forced_full"] == 2
    assert stats["per_branch"]["cond"]["max_consecutive_cached_observed"] == 2
    assert stats["per_branch"]["cond"]["max_consecutive_forced_full"] == 2
    assert [block.calls for block in model.layers] == [3, 3, 3]


@torch.no_grad()
def test_sea_cache_raw_vision_indicator_includes_conditioning_frames():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()

    def raw_vision(module, args, kwargs):
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        latent = hidden_states.permute(3, 0, 1, 2)
        return [latent]

    model.enable_cache(
        _make_config(
            runtime,
            threshold=1e-6,
            indicator_source="raw_vision_latents",
            raw_vision_callback=raw_vision,
        )
    )

    first_input = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    with model.cache_context("cond"):
        model(first_input)

    # The noisy frame stays fixed, but changing the conditioning frame changes the complete raw-latent indicator.
    runtime.update(step=1, sigma=0.9)
    second_input = torch.tensor([100.0, 2.0]).reshape(2, 1, 1, 1)
    with model.cache_context("cond"):
        model(second_input)

    stats = model.get_cache_stats()
    assert stats["indicator_source"] == "raw_vision_latents"
    assert stats["gate_trace"] == [True, True]
    assert stats["actual_full_executions"] == 2
    assert stats["actual_reuses"] == 0
    assert model.layers[0].indicator_norm.calls == 0


@torch.no_grad()
def test_sea_cache_residual_order_one_uses_actual_full_step_history():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 4}
    config = _make_config(runtime, residual_order=1, threshold=0.0)
    model = DummySeaTransformer()
    model.enable_cache(config)

    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))

    runtime.update(step=1, sigma=0.7)
    with model.cache_context("cond"):
        model(torch.full((1, 1, 1, 1), 2.0))

    config.threshold = 100.0
    runtime.update(step=2, sigma=0.5)
    with model.cache_context("cond"):
        _, output = model(torch.full((1, 1, 1, 1), 3.0))

    # Full residuals are 7 and 14 at steps 0 and 1, so linear extrapolation predicts 21 at step 2.
    torch.testing.assert_close(output, torch.full_like(output, 24.0))
    root_hook = _get_root_hook(model)
    assert root_hook.num_full_steps == 2
    assert root_hook.num_cached_steps == 1


@torch.no_grad()
def test_sea_cache_replays_gate_schedule_and_reports_natural_mismatches():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 3}
    model = DummySeaTransformer()
    model.enable_cache(
        _make_config(
            runtime,
            threshold=0.0,
            gate_schedule=(True, False, True),
        )
    )

    for step, sigma in enumerate((0.9, 0.6, 0.3)):
        runtime.update(step=step, sigma=sigma)
        with model.cache_context("cond"):
            model(torch.full((1, 1, 1, 1), float(step + 1)))

    stats = model.get_cache_stats()
    assert stats["gate_trace"] == [True, False, True]
    assert stats["gate_schedule_replayed"]
    assert stats["gate_schedule_mismatches"] == 1
    assert stats["actual_full_executions"] == 2
    assert stats["actual_reuses"] == 1


@torch.no_grad()
def test_cache_context_registry_is_refreshed_when_cache_is_enabled_after_an_uncached_call():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()

    # Pipeline cache contexts may be entered before a cache is enabled (for example, during the baseline).
    with model.cache_context("baseline"):
        model(torch.ones(1, 1, 1, 1))

    model.enable_cache(_make_config(runtime))
    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))

    assert model.get_cache_stats()["branch_full_executions"] == {"cond": 1}


@torch.no_grad()
def test_sea_cache_fails_open_without_vision_metadata():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    config = _make_config(runtime)
    config.metadata_callback = lambda module, args, kwargs: None
    model = DummySeaTransformer()
    model.enable_cache(config)

    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))
    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))

    assert [block.calls for block in model.layers] == [2, 2, 2]
    root_hook = _get_root_hook(model)
    assert root_hook.num_full_steps == 2
    assert root_hook.num_cached_steps == 0


@torch.no_grad()
def test_sea_cache_non_adjacent_steps_start_a_new_residual_trajectory():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    config = _make_config(runtime, threshold=0.0, retention_steps=0, residual_order=1)
    model = DummySeaTransformer()
    model.enable_cache(config)

    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))
    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        model(torch.full((1, 1, 1, 1), 100.0))

    config.threshold = 100.0
    runtime.update(step=0, sigma=0.9)
    with model.cache_context("cond"):
        model(torch.full((1, 1, 1, 1), 2.0))
    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        _, output = model(torch.full((1, 1, 1, 1), 2.0))

    torch.testing.assert_close(output, torch.full_like(output, 16.0))
    assert [block.calls for block in model.layers] == [3, 3, 3]
    assert model.get_cache_stats()["gate_trace"] == [True, True, True, False]


@torch.no_grad()
def test_sea_cache_fails_open_for_shape_changes():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()
    model.enable_cache(_make_config(runtime, residual_order=1))

    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))

    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        model(torch.ones(1, 2, 1, 1))

    root_hook = _get_root_hook(model)
    assert root_hook.num_full_steps == 2
    assert root_hook.num_cached_steps == 0
    assert [block.calls for block in model.layers] == [2, 2, 2]


def test_sea_cache_is_inference_only_and_fails_open_with_autograd():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()
    model.enable_cache(_make_config(runtime))

    with torch.enable_grad(), model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1, requires_grad=True))
    runtime.update(step=1, sigma=0.5)
    with torch.enable_grad(), model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1, requires_grad=True))

    root_hook = _get_root_hook(model)
    assert root_hook.shared_state.transformer_calls == 2
    assert root_hook.shared_state.fail_open_calls == 2
    assert root_hook.num_full_steps == 2
    assert root_hook.num_cached_steps == 0
    assert [block.calls for block in model.layers] == [2, 2, 2]


@torch.no_grad()
def test_sea_filter_density_normalizes_gain_to_unit_mean():
    impulse = torch.zeros(2, 3, 4, 1)
    impulse[0, 0, 0, 0] = 1

    filtered = _apply_sea_filter(impulse, sigma=0.5, power_exp=3.0)
    recovered_gain = torch.fft.fftn(filtered.float(), dim=(0, 1, 2))

    torch.testing.assert_close(
        recovered_gain.real.mean(),
        torch.tensor(1.0),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        recovered_gain.imag,
        torch.zeros_like(recovered_gain.imag),
        atol=1e-5,
        rtol=0,
    )


@pytest.mark.parametrize("sigma", [0.0, 1.0])
def test_sea_filter_is_finite_at_scheduler_endpoints(sigma):
    filtered = _apply_sea_filter(torch.randn(2, 2, 2, 4), sigma=sigma, power_exp=2.0)

    assert torch.isfinite(filtered).all()
    assert filtered.abs().sum() > 0


@torch.no_grad()
def test_sea_cache_single_block_supports_full_and_cached_execution_then_disables_cleanly():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer(num_layers=1)
    model.enable_cache(_make_config(runtime))

    with model.cache_context("cond"):
        _, first_output = model(torch.ones(1, 1, 1, 1))
    torch.testing.assert_close(first_output, torch.full_like(first_output, 2.0))

    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        _, cached_output = model(torch.full((1, 1, 1, 1), 2.0))
    torch.testing.assert_close(cached_output, torch.full_like(cached_output, 3.0))
    assert model.layers[0].calls == 1
    assert model.get_cache_stats()["actual_full_executions"] == 1
    assert model.get_cache_stats()["actual_reuses"] == 1

    model.disable_cache()

    _, output = model(torch.ones(1, 1, 1, 1))

    torch.testing.assert_close(output, torch.full_like(output, 2.0))
    assert model.layers[0].calls == 2
    assert model.layers[0]._diffusers_hook.hooks == {}


@torch.no_grad()
def test_sea_cache_fails_open_for_parameter_sharded_blocks():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()
    model.layers[0]._get_fsdp_state = lambda: object()
    model.enable_cache(_make_config(runtime))

    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))
    runtime.update(step=1, sigma=0.5)
    with model.cache_context("cond"):
        model(torch.ones(1, 1, 1, 1))

    stats = model.get_cache_stats()
    assert stats["actual_full_executions"] == 2
    assert stats["actual_reuses"] == 0
    assert stats["fail_open_calls"] == 2
    assert [block.calls for block in model.layers] == [2, 2, 2]


def test_sea_cache_failed_enable_rolls_back_only_new_hooks():
    runtime = {"step": 0, "sigma": 0.9, "num_steps": 2}
    model = DummySeaTransformer()
    existing_hook = ModelHook()
    middle_registry = HookRegistry.check_if_exists_or_initialize(model.layers[1])
    middle_registry.register_hook(existing_hook, _SEA_CACHE_BLOCK_HOOK)

    with pytest.raises(ValueError, match="already exists"):
        model.enable_cache(_make_config(runtime))

    assert not model.is_cache_enabled
    assert model._diffusers_hook.get_hook(_SEA_CACHE_ROOT_HOOK) is None
    assert model.layers[0]._diffusers_hook.get_hook(_SEA_CACHE_LEADER_BLOCK_HOOK) is None
    assert middle_registry.get_hook(_SEA_CACHE_BLOCK_HOOK) is existing_hook
    assert not hasattr(model.layers[2], "_diffusers_hook")


def test_sea_cache_config_defaults():
    config = SeaCacheConfig()

    assert config.threshold == 0.25
    assert config.residual_order == 1
    assert config.max_consecutive_cached == 2
    assert config.indicator_source == "raw_vision_latents"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": -1.0}, "threshold"),
        ({"residual_order": 2}, "residual_order"),
        ({"retention_steps": -1}, "retention_steps"),
        ({"cache_end_steps": -1}, "cache_end_steps"),
        ({"max_consecutive_cached": -1}, "max_consecutive_cached"),
        ({"max_consecutive_cached": 1.5}, "max_consecutive_cached"),
        ({"max_consecutive_cached": True}, "max_consecutive_cached"),
        ({"power_exp": 0.0}, "power_exp"),
        ({"indicator_source": "raw"}, "indicator_source"),
        ({"threshold": float("nan")}, "threshold"),
        ({"power_exp": float("inf")}, "power_exp"),
    ],
)
def test_sea_cache_config_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SeaCacheConfig(**kwargs)


def test_sea_cache_gate_schedule_validation():
    with pytest.raises(TypeError, match="gate_schedule"):
        SeaCacheConfig(gate_schedule=(True, 1))


@pytest.mark.parametrize("callback_name", ["metadata_callback", "raw_vision_callback"])
def test_sea_cache_callback_validation(callback_name):
    with pytest.raises(TypeError, match=callback_name):
        SeaCacheConfig(**{callback_name: 1})
