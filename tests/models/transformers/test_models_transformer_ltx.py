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

from diffusers import LTXVideoTransformer3DModel
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


class LTXTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LTXVideoTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, int]:
        return (512, 4)

    @property
    def input_shape(self) -> tuple[int, int]:
        return (512, 4)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "in_channels": 4,
            "out_channels": 4,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "num_layers": 1,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 2
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        embedding_dim = 16
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames * height * width, num_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length)).bool().to(torch_device),
            "num_frames": num_frames,
            "height": height,
            "width": width,
        }


class TestLTXTransformer(LTXTransformerTesterConfig, ModelTesterMixin):
    """Core model tests for LTX Video Transformer."""


class TestLTXTransformerMemory(LTXTransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for LTX Video Transformer."""


class TestLTXTransformerTraining(LTXTransformerTesterConfig, TrainingTesterMixin):
    """Training tests for LTX Video Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"LTXVideoTransformer3DModel"})


class TestLTXTransformerCompile(LTXTransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for LTX Video Transformer."""


# TODO: Add pretrained_model_name_or_path once a tiny LTX model is available on the Hub
# class TestLTXTransformerBitsAndBytes(LTXTransformerTesterConfig, BitsAndBytesTesterMixin):
#     """BitsAndBytes quantization tests for LTX Video Transformer."""


# TODO: Add pretrained_model_name_or_path once a tiny LTX model is available on the Hub
# class TestLTXTransformerTorchAo(LTXTransformerTesterConfig, TorchAoTesterMixin):
#     """TorchAo quantization tests for LTX Video Transformer."""
