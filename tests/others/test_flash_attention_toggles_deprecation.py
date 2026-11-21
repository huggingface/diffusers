# coding=utf-8
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

from unittest.mock import patch

import pytest

from diffusers import UNet2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.transformers.transformer_flux import FluxAttention


def get_model():
    return UNet2DModel(
        block_out_channels=(4, 8),
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=8,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )


def assert_warns_once(fn, name):
    with pytest.warns(FutureWarning, match=f"`set_use_{name}_flash_attention` (is|are) deprecated") as record:
        fn()
    assert len([w for w in record if w.category is FutureWarning]) == 1


# `deprecate()` raises once the library reaches the removal version, so these tests fail until the methods are removed.
@pytest.mark.parametrize("name", ["npu", "xla"])
class TestFlashAttentionTogglesDeprecation:
    def test_enable_on_model(self, name):
        model = get_model()
        with patch.object(model, f"_set_use_{name}_flash_attention") as set_use:
            assert_warns_once(getattr(model, f"enable_{name}_flash_attention"), name)
        assert set_use.call_args.args[0] is True

    def test_disable_on_model(self, name):
        model = get_model()
        assert_warns_once(getattr(model, f"disable_{name}_flash_attention"), name)

        processors = [module.processor for module in model.modules() if isinstance(module, Attention)]
        assert len(processors) > 0
        assert all(isinstance(processor, AttnProcessor2_0) for processor in processors)

    def test_set_use_on_model(self, name):
        model = get_model()
        assert_warns_once(lambda: getattr(model, f"set_use_{name}_flash_attention")(False), name)

    def test_set_use_on_attention(self, name):
        attention = Attention(query_dim=8, heads=1, dim_head=8)
        assert_warns_once(lambda: getattr(attention, f"set_use_{name}_flash_attention")(False), name)
        assert isinstance(attention.processor, AttnProcessor2_0)

    def test_set_use_on_attention_module_mixin(self, name):
        attention = FluxAttention(query_dim=8, heads=1, dim_head=8)
        with (
            patch(f"diffusers.models.attention.is_torch_{name}_available", return_value=True),
            patch.object(attention, "set_attention_backend") as set_attention_backend,
        ):
            assert_warns_once(lambda: getattr(attention, f"set_use_{name}_flash_attention")(True), name)
        set_attention_backend.assert_called_once_with(f"_native_{name}")
