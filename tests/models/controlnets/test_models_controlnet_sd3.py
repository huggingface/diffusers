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

import torch

from diffusers import SD3ControlNetModel, SD3Transformer2DModel

from ...testing_utils import enable_full_determinism


enable_full_determinism()


def get_dummy_transformer():
    torch.manual_seed(0)
    return SD3Transformer2DModel(
        sample_size=4,
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=3,
        attention_head_dim=4,
        num_attention_heads=2,
        caption_projection_dim=8,
        joint_attention_dim=8,
        pooled_projection_dim=8,
    )


class TestSD3ControlNetModelFromTransformer:
    def test_from_transformer_does_not_mutate_source_config(self):
        # Regression: `from_transformer` aliased the transformer's live config and wrote
        # ControlNet-specific values into it, so building a ControlNet silently changed the
        # source transformer's `num_layers` and added `extra_conditioning_channels`.
        transformer = get_dummy_transformer()
        config_before = dict(transformer.config)

        SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=1,
            num_extra_conditioning_channels=2,
            load_weights_from_transformer=False,
        )

        assert dict(transformer.config) == config_before, (
            "`from_transformer` must not modify the source transformer's config."
        )

    def test_from_transformer_applies_controlnet_config(self):
        transformer = get_dummy_transformer()

        controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=1,
            num_extra_conditioning_channels=2,
            load_weights_from_transformer=False,
        )

        assert controlnet.config.num_layers == 1
        assert controlnet.config.extra_conditioning_channels == 2

    def test_from_transformer_num_layers_falls_back_to_transformer(self):
        transformer = get_dummy_transformer()

        controlnet = SD3ControlNetModel.from_transformer(
            transformer,
            num_layers=None,
            num_extra_conditioning_channels=1,
            load_weights_from_transformer=False,
        )

        assert controlnet.config.num_layers == transformer.config.num_layers
