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

from diffusers import BooguImageTransformer2DModel
from diffusers.models.transformers.rope_boogu import BooguImageRotaryPosEmbed
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# Tiny config: hidden_size // num_attention_heads must equal sum(axes_dim_rope).
# Here 12 // 2 == 6 == 2 + 2 + 2.
_AXES_DIM_ROPE = (2, 2, 2)
_AXES_LENS = (16, 16, 16)
_INSTRUCTION_FEAT_DIM = 8
_THETA = 10000


class BooguImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return BooguImageTransformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return None  # No tiny Hub checkpoint yet; hub-dependent tests are skipped.

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "in_channels": 4,
            "hidden_size": 12,
            "num_layers": 2,
            "num_double_stream_layers": 1,
            "num_refiner_layers": 1,
            "num_attention_heads": 2,
            "num_kv_heads": 1,
            "multiple_of": 4,
            "norm_eps": 1e-5,
            "axes_dim_rope": _AXES_DIM_ROPE,
            "axes_lens": _AXES_LENS,
            "instruction_feature_configs": {
                "instruction_feat_dim": _INSTRUCTION_FEAT_DIM,
                "reduce_type": "mean",
                "num_instruction_feat_layers": 1,
            },
            "timestep_scale": 1.0,
        }

    def get_dummy_inputs(self, height: int = 8, width: int = 8) -> dict:
        batch_size = 1
        in_channels = 4
        instruction_len = 5
        gen = self.generator

        hidden_states = randn_tensor(
            (batch_size, in_channels, height, width), generator=gen, device=torch.device(torch_device)
        )
        timestep = torch.tensor([1.0], device=torch_device)
        instruction_hidden_states = randn_tensor(
            (batch_size, instruction_len, _INSTRUCTION_FEAT_DIM), generator=gen, device=torch.device(torch_device)
        )
        instruction_attention_mask = torch.ones(batch_size, instruction_len, dtype=torch.long, device=torch_device)
        freqs_cis = BooguImageRotaryPosEmbed.get_freqs_cis(_AXES_DIM_ROPE, _AXES_LENS, theta=_THETA)

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "instruction_hidden_states": instruction_hidden_states,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": instruction_attention_mask,
        }

    @property
    def input_shape(self) -> tuple:
        return (4, 8, 8)

    @property
    def output_shape(self) -> tuple:
        return (4, 8, 8)


class TestBooguImageTransformerModel(BooguImageTransformerTesterConfig, ModelTesterMixin):
    pass


class TestBooguImageTransformerMemory(BooguImageTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestBooguImageTransformerTorchCompile(BooguImageTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(8, 8), (8, 16), (16, 16)]

    def get_dummy_inputs(self, height: int = 8, width: int = 8) -> dict:
        return BooguImageTransformerTesterConfig.get_dummy_inputs(self, height=height, width=width)


class TestBooguImageTransformerTraining(BooguImageTransformerTesterConfig, TrainingTesterMixin):
    pass
