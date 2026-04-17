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

from diffusers import HunyuanVideoTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


# ======================== HunyuanVideo Text-to-Video ========================


class HunyuanVideoTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HunyuanVideoTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-random-hunyuanvideo"

    @property
    def pretrained_model_kwargs(self):
        return {"subfolder": "transformer"}

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 1, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (4, 1, 16, 16)

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
            "patch_size": 1,
            "patch_size_t": 1,
            "guidance_embeds": True,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 4, 4),
            "image_condition_type": None,
        }

    @property
    def torch_dtype(self):
        return None

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        pooled_projection_dim = 8
        sequence_length = 12
        dtype = self.torch_dtype

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=dtype,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(
                torch_device, dtype=dtype or torch.float32
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
                dtype=dtype,
            ),
            "pooled_projections": randn_tensor(
                (batch_size, pooled_projection_dim),
                generator=self.generator,
                device=torch_device,
                dtype=dtype,
            ),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length)).to(torch_device),
            "guidance": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(
                torch_device, dtype=dtype or torch.float32
            ),
        }


class TestHunyuanVideoTransformer(HunyuanVideoTransformerTesterConfig, ModelTesterMixin):
    pass


class TestHunyuanVideoTransformerTraining(HunyuanVideoTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HunyuanVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestHunyuanVideoTransformerCompile(HunyuanVideoTransformerTesterConfig, TorchCompileTesterMixin):
    pass


class TestHunyuanVideoTransformerBitsAndBytes(HunyuanVideoTransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for HunyuanVideo Transformer."""

    @property
    def torch_dtype(self):
        return torch.float16


class TestHunyuanVideoTransformerTorchAo(HunyuanVideoTransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for HunyuanVideo Transformer."""

    @property
    def torch_dtype(self):
        return torch.bfloat16


# ======================== HunyuanVideo Image-to-Video (Latent Concat) ========================


class HunyuanVideoI2VTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HunyuanVideoTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 1, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (8, 1, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 2 * 4 + 1,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 10,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "patch_size": 1,
            "patch_size_t": 1,
            "guidance_embeds": False,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 4, 4),
            "image_condition_type": "latent_concat",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 2 * 4 + 1
        num_frames = 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
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
        }


class TestHunyuanVideoI2VTransformer(HunyuanVideoI2VTransformerTesterConfig, ModelTesterMixin):
    def test_output(self):
        super().test_output(expected_output_shape=(1, *self.output_shape))


# ======================== HunyuanVideo Token Replace Image-to-Video ========================


class HunyuanVideoTokenReplaceTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HunyuanVideoTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple:
        return (4, 1, 16, 16)

    @property
    def input_shape(self) -> tuple:
        return (8, 1, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 2,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 10,
            "num_layers": 1,
            "num_single_layers": 1,
            "num_refiner_layers": 1,
            "patch_size": 1,
            "patch_size_t": 1,
            "guidance_embeds": True,
            "text_embed_dim": 16,
            "pooled_projection_dim": 8,
            "rope_axes_dim": (2, 4, 4),
            "image_condition_type": "token_replace",
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 2
        num_frames = 1
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
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
            "guidance": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(
                torch_device, dtype=torch.float32
            ),
        }


class TestHunyuanVideoTokenReplaceTransformer(HunyuanVideoTokenReplaceTransformerTesterConfig, ModelTesterMixin):
    def test_output(self):
        super().test_output(expected_output_shape=(1, *self.output_shape))
