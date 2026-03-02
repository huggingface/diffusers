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

from diffusers import QwenImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..test_modeling_common import LoraHotSwappingForModelTesterMixin
from ..testing_utils import (
    AttentionTesterMixin,
    ContextParallelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class QwenImageTransformerTesterConfig:
    model_class = QwenImageTransformer2DModel
    pretrained_model_name_or_path = ""
    pretrained_model_kwargs = {"subfolder": "transformer"}

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        # __init__ parameters:
        #   patch_size: int = 2
        #   in_channels: int = 64
        #   out_channels: Optional[int] = 16
        #   num_layers: int = 60
        #   attention_head_dim: int = 128
        #   num_attention_heads: int = 24
        #   joint_attention_dim: int = 3584
        #   guidance_embeds: bool = False
        #   axes_dims_rope: Tuple[int, int, int] = <complex>
        return {}

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        # forward() parameters:
        #   hidden_states: torch.Tensor
        #   encoder_hidden_states: torch.Tensor
        #   encoder_hidden_states_mask: torch.Tensor
        #   timestep: torch.LongTensor
        #   img_shapes: Optional[List[Tuple[int, int, int]]]
        #   txt_seq_lens: Optional[List[int]]
        #   guidance: torch.Tensor
        #   attention_kwargs: Optional[Dict[str, Any]]
        #   controlnet_block_samples
        #   return_dict: bool = True
        # TODO: Fill in dummy inputs
        return {}

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 1)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 1)


class TestQwenImageTransformerModel(QwenImageTransformerTesterConfig, ModelTesterMixin):
    pass


class TestQwenImageTransformerMemory(QwenImageTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestQwenImageTransformerAttention(QwenImageTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestQwenImageTransformerTorchCompile(QwenImageTransformerTesterConfig, TorchCompileTesterMixin):
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        # TODO: Implement dynamic input generation
        return {}


class TestQwenImageTransformerLora(QwenImageTransformerTesterConfig, LoraTesterMixin):
    pass


class TestQwenImageTransformerContextParallel(QwenImageTransformerTesterConfig, ContextParallelTesterMixin):
    pass


class TestQwenImageTransformerTraining(QwenImageTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestQwenImageTransformerLoraHotSwappingForModel(QwenImageTransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    different_shapes_for_compilation = [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        # TODO: Implement dynamic input generation
        return {}
