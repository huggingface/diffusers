# Copyright 2026 The AnyFlow Team, NVIDIA Corp., and The HuggingFace Team. All rights reserved.
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

from diffusers import AnyFlowTransformer3DModel
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


class AnyFlowTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AnyFlowTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, 2, 4, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, 2, 4, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool]:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 4,
            "out_channels": 4,
            "text_dim": 16,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 2,
            "cross_attn_norm": True,
            "rope_max_seq_len": 32,
            "gate_value": 0.25,
            "deltatime_type": "r",
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_frames = 2
        num_channels = 4
        height = 16
        width = 16
        text_seq_len = 12
        text_dim = 16

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_frames, num_channels, height, width),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
            "timestep": torch.full((batch_size, num_frames), 500.0, device=torch_device, dtype=self.torch_dtype),
            "r_timestep": torch.full((batch_size, num_frames), 250.0, device=torch_device, dtype=self.torch_dtype),
            "encoder_hidden_states": randn_tensor(
                (batch_size, text_seq_len, text_dim),
                generator=self.generator,
                device=torch_device,
                dtype=self.torch_dtype,
            ),
        }


class TestAnyFlowTransformer3D(AnyFlowTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for AnyFlow Transformer 3D (bidirectional variant)."""

    def test_attention_processor_api(self):
        model = self.model_class(**self.get_init_dict())
        assert len(model.attn_processors) > 0
        model.set_attn_processor(model.attn_processors)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestAnyFlowTransformer3DMemory(AnyFlowTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AnyFlow Transformer 3D."""


class TestAnyFlowTransformer3DTraining(AnyFlowTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for AnyFlow Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"AnyFlowTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestAnyFlowTransformer3DAttention(AnyFlowTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for AnyFlow Transformer 3D."""


class TestAnyFlowTransformer3DCompile(AnyFlowTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for AnyFlow Transformer 3D."""
