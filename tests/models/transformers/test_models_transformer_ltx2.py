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

import pytest
import torch

from diffusers import LTX2VideoTransformer3DModel
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


class LTX2TransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LTX2VideoTransformer3DModel

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
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "num_layers": 2,
            "qk_norm": "rms_norm_across_heads",
            "caption_channels": 16,
            "rope_double_precision": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 2
        num_frames = 2
        num_channels = 4
        height = 16
        width = 16
        audio_num_frames = 9
        audio_num_channels = 2
        num_mel_bins = 2
        embedding_dim = 16
        sequence_length = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames * height * width, num_channels),
                generator=self.generator,
                device=torch_device,
            ),
            "audio_hidden_states": randn_tensor(
                (batch_size, audio_num_frames, audio_num_channels * num_mel_bins),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "audio_encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": (randn_tensor((batch_size,), generator=self.generator, device=torch_device).abs() * 1000),
            "encoder_attention_mask": torch.ones((batch_size, sequence_length)).bool().to(torch_device),
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "audio_num_frames": audio_num_frames,
            "fps": 25.0,
        }


class TestLTX2Transformer(LTX2TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for LTX2 Video Transformer."""


class TestLTX2TransformerMemory(LTX2TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for LTX2 Video Transformer."""


class TestLTX2TransformerTraining(LTX2TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for LTX2 Video Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        super().test_gradient_checkpointing_is_applied(expected_set={"LTX2VideoTransformer3DModel"})


class TestLTX2TransformerAttention(LTX2TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for LTX2 Video Transformer."""

    @pytest.mark.skip(
        "LTX2Attention does not set is_cross_attention, so fuse_projections tries to fuse Q+K+V together even for cross-attention modules with different input dimensions."
    )
    def test_fuse_unfuse_qkv_projections(self, atol=1e-3, rtol=0):
        pass


class TestLTX2TransformerCompile(LTX2TransformerTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for LTX2 Video Transformer."""


# TODO: Add pretrained_model_name_or_path once a tiny LTX2 model is available on the Hub
# class TestLTX2TransformerBitsAndBytes(LTX2TransformerTesterConfig, BitsAndBytesTesterMixin):
#     """BitsAndBytes quantization tests for LTX2 Video Transformer."""


# TODO: Add pretrained_model_name_or_path once a tiny LTX2 model is available on the Hub
# class TestLTX2TransformerTorchAo(LTX2TransformerTesterConfig, TorchAoTesterMixin):
#     """TorchAo quantization tests for LTX2 Video Transformer."""
