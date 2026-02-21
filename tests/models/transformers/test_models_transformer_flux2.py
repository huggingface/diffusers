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

from diffusers import Flux2Transformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    ContextParallelTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    LoraHotSwappingForModelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class Flux2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return Flux2Transformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (16, 4)

    @property
    def model_split_percents(self) -> list:
        # We override the items here because the transformer under consideration is small.
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        # Skip setting testing with default: AttnProcessor
        return True

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": 1,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 16,
            "num_attention_heads": 2,
            "joint_attention_dim": 32,
            "timestep_guidance_channels": 256,  # Hardcoded in original code
            "axes_dims_rope": [4, 4, 4, 4],
        }

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)  # [height * width, 4]
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2Transformer(Flux2TransformerTesterConfig, ModelTesterMixin):
    pass


class TestFlux2TransformerMemory(Flux2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Flux2 Transformer."""


class TestFlux2TransformerTraining(Flux2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Flux2 Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Flux2Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestFlux2TransformerAttention(Flux2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Flux2 Transformer."""


class TestFlux2TransformerContextParallel(Flux2TransformerTesterConfig, ContextParallelTesterMixin):
    """Context Parallel inference tests for Flux2 Transformer."""


class TestFlux2TransformerLoRA(Flux2TransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for Flux2 Transformer."""


class TestFlux2TransformerLoRAHotSwap(Flux2TransformerTesterConfig, LoraHotSwappingForModelTesterMixin):
    """LoRA hot-swapping tests for Flux2 Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for LoRA hotswap tests."""
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerCompile(Flux2TransformerTesterConfig, TorchCompileTesterMixin):
    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict[str, torch.Tensor]:
        """Override to support dynamic height/width for compilation tests."""
        batch_size = 1
        num_latent_channels = 4
        sequence_length = 48
        embedding_dim = 32

        hidden_states = randn_tensor(
            (batch_size, height * width, num_latent_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
        )

        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        guidance = torch.tensor([1.0]).to(torch_device).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerBitsAndBytes(Flux2TransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Flux2 Transformer."""


class TestFlux2TransformerTorchAo(Flux2TransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Flux2 Transformer."""


class TestFlux2TransformerGGUF(Flux2TransformerTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Flux2 Transformer."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/unsloth/FLUX.2-dev-GGUF/blob/main/flux2-dev-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real FLUX2 model dimensions.

        Flux2 defaults: in_channels=128, joint_attention_dim=15360
        """
        batch_size = 1
        height = 64
        width = 64
        sequence_length = 512

        hidden_states = randn_tensor(
            (batch_size, height * width, 128), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, 15360), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )

        # Flux2 uses 4D image/text IDs (t, h, w, l)
        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device, self.torch_dtype)
        guidance = torch.tensor([3.5]).to(torch_device, self.torch_dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }


class TestFlux2TransformerGGUFCompile(Flux2TransformerTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Flux2 Transformer."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/unsloth/FLUX.2-dev-GGUF/blob/main/flux2-dev-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real FLUX2 model dimensions.

        Flux2 defaults: in_channels=128, joint_attention_dim=15360
        """
        batch_size = 1
        height = 64
        width = 64
        sequence_length = 512

        hidden_states = randn_tensor(
            (batch_size, height * width, 128), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, 15360), generator=self.generator, device=torch_device, dtype=self.torch_dtype
        )

        # Flux2 uses 4D image/text IDs (t, h, w, l)
        t_coords = torch.arange(1)
        h_coords = torch.arange(height)
        w_coords = torch.arange(width)
        l_coords = torch.arange(1)
        image_ids = torch.cartesian_prod(t_coords, h_coords, w_coords, l_coords)
        image_ids = image_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        text_t_coords = torch.arange(1)
        text_h_coords = torch.arange(1)
        text_w_coords = torch.arange(1)
        text_l_coords = torch.arange(sequence_length)
        text_ids = torch.cartesian_prod(text_t_coords, text_h_coords, text_w_coords, text_l_coords)
        text_ids = text_ids.unsqueeze(0).expand(batch_size, -1, -1).to(torch_device)

        timestep = torch.tensor([1.0]).to(torch_device, self.torch_dtype)
        guidance = torch.tensor([3.5]).to(torch_device, self.torch_dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "img_ids": image_ids,
            "txt_ids": text_ids,
            "timestep": timestep,
            "guidance": guidance,
        }
