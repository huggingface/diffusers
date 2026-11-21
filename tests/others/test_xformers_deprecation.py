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

import pytest

from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.utils.import_utils import is_xformers_available


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


def get_pipeline():
    return DDPMPipeline(unet=get_model(), scheduler=DDPMScheduler())


def assert_warns_once(fn):
    with pytest.warns(FutureWarning, match="xFormers memory-efficient attention is deprecated") as record:
        fn()
    assert len([w for w in record if w.category is FutureWarning]) == 1


# `deprecate()` raises once the library reaches the removal version, so these tests fail until the methods are removed.
class TestXFormersDeprecation:
    @pytest.mark.parametrize("get_target", [get_model, get_pipeline], ids=["model", "pipeline"])
    def test_disable(self, get_target):
        target = get_target()
        assert_warns_once(target.disable_xformers_memory_efficient_attention)

        model = target.unet if isinstance(target, DDPMPipeline) else target
        processors = [module.processor for module in model.modules() if isinstance(module, Attention)]
        assert len(processors) > 0
        assert all(isinstance(processor, AttnProcessor2_0) for processor in processors)

    @pytest.mark.parametrize("get_target", [get_model, get_pipeline], ids=["model", "pipeline"])
    def test_set_use(self, get_target):
        target = get_target()
        assert_warns_once(lambda: target.set_use_memory_efficient_attention_xformers(False))

    @pytest.mark.parametrize("attention_class", [Attention, FluxAttention])
    def test_set_use_on_attention_module(self, attention_class):
        attention = attention_class(query_dim=8, heads=1, dim_head=8)
        assert_warns_once(lambda: attention.set_use_memory_efficient_attention_xformers(False))

    @pytest.mark.skipif(is_xformers_available(), reason="Checks the error raised when xFormers is not installed.")
    @pytest.mark.parametrize("get_target", [get_model, get_pipeline], ids=["model", "pipeline"])
    def test_enable_without_xformers(self, get_target):
        target = get_target()

        def enable():
            with pytest.raises(ModuleNotFoundError):
                target.enable_xformers_memory_efficient_attention()

        assert_warns_once(enable)
