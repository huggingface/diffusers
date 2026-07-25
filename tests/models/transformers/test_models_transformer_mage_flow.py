# coding=utf-8
# Copyright 2025 The Mage Team and HuggingFace Inc.
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

from diffusers import MageFlowTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class MageFlowTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return MageFlowTransformer2DModel

    @property
    def pretrained_model_name_or_path(self):
        return ""  # TODO: Set Hub repository ID

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "in_channels": 4,
            "out_channels": 4,
            "context_in_dim": 32,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_layers": 2,
            "axes_dim": [4, 6, 6],
            "patch_size": 1,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 2
        img_seq_len = 4  # 2x2 spatial
        txt_seq_len = 3

        hidden_states = randn_tensor(
            (batch_size, img_seq_len, 4),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, txt_seq_len, 32),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        timestep = torch.tensor([0.5, 0.8], device=torch_device, dtype=self.torch_dtype)

        img_ids = torch.zeros(img_seq_len, 3, device=torch_device, dtype=self.torch_dtype)
        img_ids[:, 1] = torch.arange(2, device=torch_device, dtype=self.torch_dtype).repeat_interleave(2)
        img_ids[:, 2] = torch.arange(2, device=torch_device, dtype=self.torch_dtype).repeat(2)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
        }

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (2, 4, 4)  # (batch, seq_len, channels)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (2, 4, 4)


class TestMageFlowTransformerModel(MageFlowTransformerTesterConfig, ModelTesterMixin):
    # The fixed-size RoPE frequency buffers (4096 × axes_dim) dominate the tiny
    # test model's parameter count, making it impossible to split the model across
    # GPUs at the granularity accelerate uses. Skip multi-GPU parallelism for the
    # tiny test config; real-size models split correctly.
    test_model_parallelism = None


class TestMageFlowTransformerMemory(MageFlowTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestMageFlowTransformerTorchCompile(MageFlowTransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(2, 2), (2, 4), (4, 4)]

    def get_dummy_inputs(self, height: int = 2, width: int = 2) -> dict[str, torch.Tensor]:
        batch_size = 2
        img_seq_len = height * width
        txt_seq_len = 3

        hidden_states = randn_tensor(
            (batch_size, img_seq_len, 4),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, txt_seq_len, 32),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )
        timestep = torch.tensor([0.5, 0.8], device=torch_device, dtype=self.torch_dtype)

        img_ids = torch.zeros(img_seq_len, 3, device=torch_device, dtype=self.torch_dtype)
        img_ids[:, 1] = torch.arange(height, device=torch_device, dtype=self.torch_dtype).repeat_interleave(width)
        img_ids[:, 2] = torch.arange(width, device=torch_device, dtype=self.torch_dtype).repeat(height)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "img_ids": img_ids,
        }


class TestMageFlowTransformerTraining(MageFlowTransformerTesterConfig, TrainingTesterMixin):
    pass


class TestMageFlowTransformerAttention(MageFlowTransformerTesterConfig, AttentionTesterMixin):
    pass
