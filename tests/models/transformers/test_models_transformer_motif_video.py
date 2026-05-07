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
    BaseModelTesterConfig,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class MotifVideoTransformerTesterConfig(BaseModelTesterConfig):
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

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 33, 9, 16, 16)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 16, 9, 16, 16)

    def get_init_dict(self) -> dict[str, int | list[int] | float | str | bool]:
        return {
            "in_channels": 33,
            "out_channels": 16,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_decoder_layers": 0,
            "mlp_ratio": 4.0,
            "patch_size": 1,
            "patch_size_t": 1,
            "qk_norm": "rms_norm",
            "norm_type": "layer_norm",
            "text_embed_dim": 32,
            "image_embed_dim": 4,
            "rope_theta": 256.0,
            "rope_axes_dim": (4, 4, 4),
            "enable_text_cross_attention_dual": False,
            "enable_text_cross_attention_single": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 33
        num_frames = 9
        height = 16
        width = 16
        text_embed_dim = 32
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_embed_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestMotifVideoTransformerModel(MotifVideoTransformerTesterConfig, ModelTesterMixin):
    pass


class TestMotifVideoTransformerMemory(MotifVideoTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestMotifVideoTransformerTorchCompile(MotifVideoTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 33
        num_frames = 9
        text_embed_dim = 32
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_embed_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestMotifVideoTransformerLora(MotifVideoTransformerTesterConfig, LoraTesterMixin):
    pass


class TestMotifVideoTransformerTraining(MotifVideoTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestMotifVideoTransformerAttention(MotifVideoTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestMotifVideoTransformerLoraHotSwappingForModel(
    MotifVideoTransformerTesterConfig, LoraHotSwappingForModelTesterMixin
):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 33
        num_frames = 9
        text_embed_dim = 32
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_embed_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }
