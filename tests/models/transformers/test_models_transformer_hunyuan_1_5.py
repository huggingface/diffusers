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

from diffusers import HunyuanVideo15Transformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HunyuanVideo15TransformerTesterConfig(BaseModelTesterConfig):
    text_embed_dim = 16
    text_embed_2_dim = 8
    image_embed_dim = 12

    @property
    def model_class(self):
        return HunyuanVideo15Transformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.99, 0.99, 0.99]

    @property
    def output_shape(self) -> tuple:
        return (4, 1, 8, 8)

    @property
    def input_shape(self) -> tuple:
        return (4, 1, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "num_layers": 2,
            "num_refiner_layers": 1,
            "mlp_ratio": 2.0,
            "patch_size": 1,
            "patch_size_t": 1,
            "text_embed_dim": self.text_embed_dim,
            "text_embed_2_dim": self.text_embed_2_dim,
            "image_embed_dim": self.image_embed_dim,
            "rope_axes_dim": (2, 2, 4),
            "target_size": 16,
            "task_type": "t2v",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = 8
        width = 8
        sequence_length = 6
        sequence_length_2 = 4
        image_sequence_length = 3

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([1.0]).to(torch_device).expand(batch_size),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, self.text_embed_dim), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states_2": randn_tensor(
                (batch_size, sequence_length_2, self.text_embed_2_dim), generator=self.generator, device=torch_device
            ),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length), device=torch_device),
            "encoder_attention_mask_2": torch.ones((batch_size, sequence_length_2), device=torch_device),
            "image_embeds": torch.zeros(
                (batch_size, image_sequence_length, self.image_embed_dim), device=torch_device
            ),
        }


class TestHunyuanVideo15Transformer(HunyuanVideo15TransformerTesterConfig, ModelTesterMixin):
    pass


class TestHunyuanVideo15TransformerTraining(HunyuanVideo15TransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideo15Transformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
