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

from diffusers import HunyuanVideoFramepackTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class HunyuanVideoFramepackTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HunyuanVideoFramepackTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_split_percents(self) -> list:
        return [0.5, 0.7, 0.9]

    @property
    def output_shape(self) -> tuple:
        return (4, 3, 4, 4)

    @property
    def input_shape(self) -> tuple:
        return (4, 3, 4, 4)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 10,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "patch_size": 2,
            "patch_size_t": 1,
            "guidance_embeds": True,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 4, 4),
            "image_condition_type": None,
            "has_image_proj": True,
            "image_proj_dim": 16,
            "has_clean_x_embedder": True,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 3
        height = 4
        width = 4
        text_encoder_embedding_dim = 16
        image_encoder_embedding_dim = 16
        pooled_projection_dim = 8
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "pooled_projections": randn_tensor(
                (batch_size, pooled_projection_dim), generator=self.generator, device=torch_device
            ),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length)).to(torch_device),
            "guidance": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "image_embeds": randn_tensor(
                (batch_size, sequence_length, image_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "indices_latents": torch.ones((num_frames,)).to(torch_device),
            "latents_clean": randn_tensor(
                (batch_size, num_channels, num_frames - 1, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "indices_latents_clean": torch.ones((num_frames - 1,)).to(torch_device),
            "latents_history_2x": randn_tensor(
                (batch_size, num_channels, num_frames - 1, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "indices_latents_history_2x": torch.ones((num_frames - 1,)).to(torch_device),
            "latents_history_4x": randn_tensor(
                (batch_size, num_channels, (num_frames - 1) * 4, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "indices_latents_history_4x": torch.ones(((num_frames - 1) * 4,)).to(torch_device),
        }


class TestHunyuanVideoFramepackTransformer(HunyuanVideoFramepackTransformerTesterConfig, ModelTesterMixin):
    pass


class TestHunyuanVideoFramepackTransformerTraining(HunyuanVideoFramepackTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideoFramepackTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
