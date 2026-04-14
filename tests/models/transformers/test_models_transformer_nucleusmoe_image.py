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

from diffusers import NucleusMoEImageTransformer2DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    LoraHotSwappingForModelTesterMixin,
    LoraTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class NucleusMoEImageTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return NucleusMoEImageTransformer2DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (16, 16)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (16, 16)

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 2,
            "attention_head_dim": 16,
            "num_attention_heads": 4,
            "joint_attention_dim": 16,
            "axes_dims_rope": (8, 4, 4),
            "moe_enabled": False,
            "capacity_factors": [8.0, 8.0],
        }

    def get_dummy_inputs(self) -> dict:
        batch_size = 1
        in_channels = 16
        joint_attention_dim = 16
        height = width = 4
        sequence_length = 8

        hidden_states = randn_tensor(
            (batch_size, height * width, in_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, joint_attention_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length), dtype=torch.long).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        img_shapes = [(1, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }


class TestNucleusMoEImageTransformer(NucleusMoEImageTransformerTesterConfig, ModelTesterMixin):
    def test_with_attention_mask(self):
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        # Mask out some text tokens
        mask = inputs["encoder_hidden_states_mask"].clone()
        mask[:, 4:] = 0
        inputs["encoder_hidden_states_mask"] = mask

        with torch.no_grad():
            output = model(**inputs)

        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]

    def test_without_attention_mask(self):
        init_dict = self.get_init_dict()
        inputs = self.get_dummy_inputs()
        model = self.model_class(**init_dict).to(torch_device)

        inputs["encoder_hidden_states_mask"] = None

        with torch.no_grad():
            output = model(**inputs)

        assert output.sample.shape[1] == inputs["hidden_states"].shape[1]


class TestNucleusMoEImageTransformerMemory(NucleusMoEImageTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for NucleusMoE Image Transformer."""


class TestNucleusMoEImageTransformerTraining(NucleusMoEImageTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for NucleusMoE Image Transformer."""


class TestNucleusMoEImageTransformerAttention(NucleusMoEImageTransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for NucleusMoE Image Transformer."""


class TestNucleusMoEImageTransformerLoRA(NucleusMoEImageTransformerTesterConfig, LoraTesterMixin):
    """LoRA adapter tests for NucleusMoE Image Transformer."""


class TestNucleusMoEImageTransformerLoRAHotSwap(
    NucleusMoEImageTransformerTesterConfig, LoraHotSwappingForModelTesterMixin
):
    """LoRA hot-swapping tests for NucleusMoE Image Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict:
        batch_size = 1
        in_channels = 16
        joint_attention_dim = 16
        sequence_length = 8

        hidden_states = randn_tensor(
            (batch_size, height * width, in_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, joint_attention_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length), dtype=torch.long).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        img_shapes = [(1, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }


class TestNucleusMoEImageTransformerCompile(NucleusMoEImageTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for NucleusMoE Image Transformer."""

    @property
    def different_shapes_for_compilation(self):
        return [(4, 4), (4, 8), (8, 8)]

    def get_dummy_inputs(self, height: int = 4, width: int = 4) -> dict:
        batch_size = 1
        in_channels = 16
        joint_attention_dim = 16
        sequence_length = 8

        hidden_states = randn_tensor(
            (batch_size, height * width, in_channels), generator=self.generator, device=torch_device
        )
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, joint_attention_dim), generator=self.generator, device=torch_device
        )
        encoder_hidden_states_mask = torch.ones((batch_size, sequence_length), dtype=torch.long).to(torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        img_shapes = [(1, height, width)] * batch_size

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "timestep": timestep,
            "img_shapes": img_shapes,
        }


class TestNucleusMoEImageTransformerBitsAndBytes(NucleusMoEImageTransformerTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for NucleusMoE Image Transformer."""


class TestNucleusMoEImageTransformerTorchAo(NucleusMoEImageTransformerTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for NucleusMoE Image Transformer."""
