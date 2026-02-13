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

from diffusers import WanTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    BitsAndBytesTesterMixin,
    GGUFCompileTesterMixin,
    GGUFTesterMixin,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchAoTesterMixin,
    TorchCompileTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class WanTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan22-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (4, 2, 16, 16)

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
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestWanTransformer3D(WanTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Wan Transformer 3D."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestWanTransformer3DMemory(WanTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Wan Transformer 3D."""


class TestWanTransformer3DTraining(WanTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Wan Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestWanTransformer3DAttention(WanTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Wan Transformer 3D."""


class TestWanTransformer3DCompile(WanTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Wan Transformer 3D."""


class TestWanTransformer3DBitsAndBytes(WanTransformer3DTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.float16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DTorchAo(WanTransformer3DTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Wan Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUF(WanTransformer3DTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanTransformer3DGGUFCompile(WanTransformer3DTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Wan Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF/blob/main/LowNoise/Wan2.2-I2V-A14B-LowNoise-Q2_K.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def _create_quantized_model(self, config_kwargs=None, **extra_kwargs):
        return super()._create_quantized_model(
            config_kwargs, config="Wan-AI/Wan2.2-I2V-A14B-Diffusers", subfolder="transformer", **extra_kwargs
        )

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan I2V model dimensions.

        Wan 2.2 I2V: in_channels=36, text_dim=4096
        """
        return {
            "hidden_states": randn_tensor(
                (1, 36, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }
