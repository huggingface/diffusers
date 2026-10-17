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
import torch

from diffusers import TripoSplatGaussianDecoder
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class TripoSplatGaussianDecoderTesterConfig:
    main_input_name = "hidden_states"

    @property
    def model_class(self):
        return TripoSplatGaussianDecoder

    @property
    def pretrained_model_name_or_path(self):
        return None

    @property
    def pretrained_model_kwargs(self):
        return {}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "model_channels": 32,
            "cond_channels": 16,
            "num_octree_blocks": 1,
            "num_gaussian_blocks": 1,
            "num_heads": 2,
            "mlp_ratio": 2.0,
            "gaussians_per_point": 32,
            "max_voxel_level": 2,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "hidden_states": randn_tensor((1, 16, 16), generator=self.generator, device=torch_device),
            "num_gaussians": 64,
            "generator": self.generator,
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (64, 14)


class TestTripoSplatGaussianDecoderModel(TripoSplatGaussianDecoderTesterConfig, ModelTesterMixin):
    pass


class TestTripoSplatGaussianDecoderMemory(TripoSplatGaussianDecoderTesterConfig, MemoryTesterMixin):
    def get_dummy_inputs(self):
        inputs = super().get_dummy_inputs()
        inputs["generator"] = torch.manual_seed(0)
        return inputs


class TestTripoSplatGaussianDecoderTorchCompile(TripoSplatGaussianDecoderTesterConfig, TorchCompileTesterMixin):
    @pytest.mark.skip(reason="Adaptive octree sampling uses data-dependent tensor sizes; compile the repeated blocks.")
    def test_torch_compile_recompilation_and_graph_break(self):
        pass

    @pytest.mark.skip(reason="Adaptive octree sampling cannot be exported as one static graph.")
    def test_compile_works_with_aot(self, tmp_path):
        pass

    @pytest.mark.skip(reason="Adaptive octree sampling uses data-dependent tensor sizes; compile the repeated blocks.")
    def test_compile_on_different_shapes(self):
        pass


class TestTripoSplatGaussianDecoderAttention(TripoSplatGaussianDecoderTesterConfig, AttentionTesterMixin):
    pass
