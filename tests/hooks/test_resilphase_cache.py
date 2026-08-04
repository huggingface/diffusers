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

from diffusers import ResilPhaseCacheConfig, apply_resilphase_cache
from diffusers.hooks import HookRegistry
from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
from diffusers.hooks.resilphase_cache import ResilPhaseState


class DummyBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
        self.calls += 1
        return hidden_states + 1, encoder_hidden_states + 2


class DummyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([DummyBlock(), DummyBlock(), DummyBlock()])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        controlnet_block_samples: torch.Tensor | None = None,
        controlnet_single_block_samples: torch.Tensor | None = None,
    ):
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states)
            if controlnet_block_samples is not None:
                hidden_states = hidden_states + controlnet_block_samples
        return hidden_states, encoder_hidden_states


@pytest.fixture(autouse=True)
def register_dummy_block():
    TransformerBlockRegistry.register(
        DummyBlock,
        TransformerBlockMetadata(return_hidden_states_index=0, return_encoder_hidden_states_index=1),
    )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"cache_interval": 0}, "cache_interval"),
        ({"warmup_steps": -1}, "warmup_steps"),
        ({"max_order": -1}, "max_order"),
        ({"mapping_method": "linear"}, "mapping_method"),
        ({"balance_alpha": 0}, "balance_alpha"),
    ],
)
def test_resilphase_config_validation(kwargs, error):
    with pytest.raises(ValueError, match=error):
        ResilPhaseCacheConfig(**kwargs)


@pytest.mark.parametrize(
    ("mapping_method", "expected_hidden_states", "expected_encoder_hidden_states"),
    [
        ("balanced", 4.670691622065688, 9.341383244131376),
        ("chebyshev", 5.0, 10.0),
    ],
)
def test_resilphase_barycentric_prediction(mapping_method, expected_hidden_states, expected_encoder_hidden_states):
    state = ResilPhaseState(ResilPhaseCacheConfig(mapping_method=mapping_method))
    state.step_index = 4
    state.history_steps = [0, 3]
    state.history_residuals = [
        (torch.tensor([1.0]), torch.tensor([2.0])),
        (torch.tensor([4.0]), torch.tensor([8.0])),
    ]

    hidden_states, encoder_hidden_states = state.predict()

    assert torch.allclose(hidden_states, torch.tensor([expected_hidden_states]))
    assert torch.allclose(encoder_hidden_states, torch.tensor([expected_encoder_hidden_states]))


@pytest.mark.parametrize("mapping_method", ["balanced", "chebyshev"])
def test_resilphase_skips_blocks_and_predicts_both_streams(mapping_method):
    model = DummyTransformer()
    config = ResilPhaseCacheConfig(
        cache_interval=3,
        warmup_steps=2,
        max_order=1,
        mapping_method=mapping_method,
    )
    apply_resilphase_cache(model, config)
    HookRegistry.check_if_exists_or_initialize(model)._set_context("cond")

    hidden_states = torch.tensor([[[0.0]]])
    encoder_hidden_states = torch.tensor([[[0.0]]])

    output_0 = model(hidden_states, encoder_hidden_states)
    output_1 = model(hidden_states + 1, encoder_hidden_states + 1)
    output_2 = model(hidden_states + 2, encoder_hidden_states + 2)

    assert torch.allclose(output_0[0], torch.tensor([[[3.0]]]))
    assert torch.allclose(output_0[1], torch.tensor([[[6.0]]]))
    assert torch.allclose(output_1[0], torch.tensor([[[4.0]]]))
    assert torch.allclose(output_1[1], torch.tensor([[[7.0]]]))
    assert torch.allclose(output_2[0], torch.tensor([[[5.0]]]))
    assert torch.allclose(output_2[1], torch.tensor([[[8.0]]]))
    assert [block.calls for block in model.transformer_blocks] == [2, 2, 2]


def test_resilphase_contexts_keep_independent_state():
    model = DummyTransformer()
    apply_resilphase_cache(model, ResilPhaseCacheConfig(cache_interval=3, warmup_steps=1, max_order=0))
    registry = HookRegistry.check_if_exists_or_initialize(model)

    hidden_states = torch.tensor([[[1.0]]])
    encoder_hidden_states = torch.tensor([[[1.0]]])

    registry._set_context("cond")
    cond_output = model(hidden_states, encoder_hidden_states)
    registry._set_context("uncond")
    uncond_output = model(hidden_states, encoder_hidden_states)

    assert torch.equal(cond_output[0], uncond_output[0])
    assert torch.equal(cond_output[1], uncond_output[1])
    assert [block.calls for block in model.transformer_blocks] == [2, 2, 2]


def test_resilphase_refreshes_after_cache_interval():
    model = DummyTransformer()
    apply_resilphase_cache(model, ResilPhaseCacheConfig(cache_interval=3, warmup_steps=2, max_order=1))
    HookRegistry.check_if_exists_or_initialize(model)._set_context("cond")

    hidden_states = torch.tensor([[[0.0]]])
    encoder_hidden_states = torch.tensor([[[0.0]]])
    for _ in range(6):
        model(hidden_states, encoder_hidden_states)

    assert [block.calls for block in model.transformer_blocks] == [3, 3, 3]


def test_resilphase_bypasses_cache_for_controlnet_residuals():
    model = DummyTransformer()
    apply_resilphase_cache(model, ResilPhaseCacheConfig(cache_interval=3, warmup_steps=1, max_order=0))
    HookRegistry.check_if_exists_or_initialize(model)._set_context("cond")

    hidden_states = torch.tensor([[[0.0]]])
    encoder_hidden_states = torch.tensor([[[0.0]]])
    model(hidden_states, encoder_hidden_states)
    model(hidden_states, encoder_hidden_states)
    output = model(
        hidden_states,
        encoder_hidden_states,
        controlnet_block_samples=torch.tensor([[[10.0]]]),
    )

    assert torch.equal(output[0], torch.tensor([[[33.0]]]))
    assert torch.equal(output[1], torch.tensor([[[6.0]]]))
    assert [block.calls for block in model.transformer_blocks] == [2, 2, 2]
