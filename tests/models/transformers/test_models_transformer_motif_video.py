# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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

from diffusers import MotifVideoTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import LoraHotSwappingForModelTesterMixin
from ..testing_utils import (
    AttentionTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class MotifVideoTransformerTesterConfig:
    @property
    def model_class(self):
        return MotifVideoTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return ""  # TODO: Set Hub repository ID

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        # __init__ parameters:
        #   in_channels: int = 33
        #   out_channels: int = 16
        #   num_attention_heads: int = 24
        #   attention_head_dim: int = 128
        #   num_layers: int = 20
        #   num_single_layers: int = 40
        #   num_decoder_layers: int = 0
        #   mlp_ratio: float = 4.0
        #   patch_size: int = 2
        #   patch_size_t: int = 1
        #   qk_norm: str = rms_norm
        #   norm_type: str = layer_norm
        #   text_embed_dim: int = 4096
        #   image_embed_dim: int | None
        #   rope_theta: float = 256.0
        #   rope_axes_dim: Tuple[int, Ellipsis] = <complex>
        #   enable_text_cross_attention_dual: bool = False
        #   enable_text_cross_attention_single: bool = False
        return {}

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        # forward() parameters:
        #   hidden_states: torch.Tensor
        #   timestep: torch.LongTensor
        #   encoder_hidden_states: torch.Tensor
        #   encoder_attention_mask: torch.Tensor | None
        #   image_embeds: torch.Tensor | None
        #   attention_kwargs: Optional[Dict[str, Any]]
        #   return_dict: bool = True
        # TODO: Fill in dummy inputs
        return {}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 1)


class TestMotifVideoTransformerModel(MotifVideoTransformerTesterConfig, ModelTesterMixin):
    pass


class TestMotifVideoTransformerMemory(MotifVideoTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestMotifVideoTransformerTorchCompile(MotifVideoTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        # TODO: Implement dynamic input generation
        return {}


class TestMotifVideoTransformerLora(MotifVideoTransformerTesterConfig, LoraTesterMixin):
    pass


class TestMotifVideoTransformerTraining(MotifVideoTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestMotifVideoTransformerAttention(MotifVideoTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestMotifVideoTransformerLoraHotSwappingForModel(MotifVideoTransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        # TODO: Implement dynamic input generation
        return {}
