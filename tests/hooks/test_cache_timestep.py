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

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from diffusers import CogVideoXTransformer3DModel, FasterCacheConfig, PyramidAttentionBroadcastConfig


@pytest.fixture
def model():
    torch.manual_seed(0)
    return CogVideoXTransformer3DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        time_embed_dim=2,
        text_embed_dim=8,
        num_layers=1,
        sample_width=8,
        sample_height=8,
        sample_frames=1,
        patch_size=2,
        temporal_compression_ratio=4,
        max_text_seq_length=8,
    ).eval()


@pytest.fixture(params=[FasterCacheConfig, PyramidAttentionBroadcastConfig])
def config_kwargs(request):
    kwargs = {"spatial_attention_block_skip_range": 2}
    if request.param is FasterCacheConfig:
        kwargs["tensor_format"] = "BFCHW"
    return request.param, kwargs


@pytest.fixture
def inputs():
    generator = torch.Generator().manual_seed(0)
    return [
        {
            "hidden_states": torch.randn(2, 1, 4, 8, 8, generator=generator),
            "encoder_hidden_states": torch.randn(2, 8, 8, generator=generator),
            "timestep": torch.full((2,), timestep),
        }
        for timestep in [900, 600, 500, 400, 200, 0]
    ]


@pytest.mark.parametrize("context_name", [None, "cond"])
@torch.no_grad()
def test_deprecated_timestep_callback_matches_cache_context(model, config_kwargs, inputs, context_name):
    config_class, kwargs = config_kwargs
    model.enable_cache(config_class(**kwargs))
    expected = []
    for step_inputs in inputs:
        with model.cache_context("cond", timestep=step_inputs["timestep"][0]):
            expected.append(model(**step_inputs).sample)
    model.disable_cache()

    timestep = None
    with pytest.warns(FutureWarning, match="current_timestep_callback.*0.45.0"):
        config = config_class(**kwargs, current_timestep_callback=lambda: timestep)
    model.enable_cache(config)

    for _ in range(2):
        for step_inputs, expected_output in zip(inputs, expected):
            timestep = step_inputs["timestep"][0]
            context = model.cache_context(context_name) if context_name is not None else nullcontext()
            with context:
                output = model(**step_inputs).sample
            torch.testing.assert_close(output, expected_output)
        model._reset_stateful_cache()


@torch.no_grad()
def test_context_timestep_takes_precedence_over_callback(model, config_kwargs, inputs):
    config_class, kwargs = config_kwargs
    callback = Mock(side_effect=AssertionError)
    with pytest.warns(FutureWarning, match="current_timestep_callback"):
        config = config_class(**kwargs, current_timestep_callback=callback)
    model.enable_cache(config)

    for step_inputs in inputs:
        with model.cache_context("cond", timestep=step_inputs["timestep"][0]):
            model(**step_inputs)
    callback.assert_not_called()


@pytest.mark.parametrize("context_name", [None, "cond"])
def test_timestep_is_required_without_callback(model, config_kwargs, inputs, context_name):
    config_class, kwargs = config_kwargs
    model.enable_cache(config_class(**kwargs))
    context = model.cache_context(context_name) if context_name is not None else nullcontext()
    with context, pytest.raises(ValueError, match="cache_context"):
        model(**inputs[0])
