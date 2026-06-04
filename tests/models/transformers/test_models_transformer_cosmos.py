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

from diffusers import CosmosTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class CosmosTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return CosmosTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 1, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 1, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list | tuple | float | bool | str]:
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "mlp_ratio": 2,
            "text_embed_dim": 16,
            "adaln_lora_dim": 4,
            "max_size": (4, 32, 32),
            "patch_size": (1, 2, 2),
            "rope_scale": (2.0, 1.0, 1.0),
            "concat_padding_mask": True,
            "extra_pos_embed_type": "learnable",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_embed_dim), generator=self.generator, device=torch_device
            ),
            "attention_mask": torch.ones((batch_size, sequence_length)).to(torch_device),
            "fps": 30,
            "padding_mask": torch.zeros(batch_size, 1, height, width).to(torch_device),
        }


class TestCosmosTransformer(CosmosTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for Cosmos Transformer."""


class TestCosmosTransformerMemory(CosmosTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Cosmos Transformer."""


class TestCosmosTransformerTraining(CosmosTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Cosmos Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class CosmosTransformerVideoToWorldTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return CosmosTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 1, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 1, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list | tuple | float | bool | str]:
        return {
            "in_channels": 4 + 1,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "mlp_ratio": 2,
            "text_embed_dim": 16,
            "adaln_lora_dim": 4,
            "max_size": (4, 32, 32),
            "patch_size": (1, 2, 2),
            "rope_scale": (2.0, 1.0, 1.0),
            "concat_padding_mask": True,
            "extra_pos_embed_type": "learnable",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_embed_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_embed_dim), generator=self.generator, device=torch_device
            ),
            "attention_mask": torch.ones((batch_size, sequence_length)).to(torch_device),
            "fps": 30,
            "condition_mask": torch.ones(batch_size, 1, num_frames, height, width).to(torch_device),
            "padding_mask": torch.zeros(batch_size, 1, height, width).to(torch_device),
        }


class TestCosmosTransformerVideoToWorld(CosmosTransformerVideoToWorldTesterConfig, ModelTesterMixin):
    """Core model tests for Cosmos Transformer (Video-to-World)."""


class TestCosmosTransformerVideoToWorldMemory(CosmosTransformerVideoToWorldTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Cosmos Transformer (Video-to-World)."""


class TestCosmosTransformerVideoToWorldTraining(CosmosTransformerVideoToWorldTesterConfig, TrainingTesterMixin):
    """Training tests for Cosmos Transformer (Video-to-World)."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"CosmosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
