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

from diffusers import WanVACETransformer3DModel
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


class WanVACETransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return WanVACETransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-wan-vace-transformer"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 2, 16, 16)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 2, 16, 16)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int] | tuple | str | bool | None]:
        return {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 32,
            "freq_dim": 256,
            "ffn_dim": 32,
            "num_layers": 4,
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
            "vace_layers": [0, 2],
            "vace_in_channels": 48,  # 3 * in_channels = 3 * 16 = 48
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 16
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 32
        sequence_length = 12

        # VACE requires control_hidden_states with vace_in_channels (3 * in_channels)
        vace_in_channels = 48

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
            "control_hidden_states": randn_tensor(
                (batch_size, vace_in_channels, num_frames, height, width),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestWanVACETransformer3D(WanVACETransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Wan VACE Transformer 3D."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")

    def test_model_parallelism(self, tmp_path):
        # Skip: Device mismatch between cuda:0 and cuda:1 in VACE control flow
        pytest.skip("Model parallelism not yet supported for WanVACE")


class TestWanVACETransformer3DMemory(WanVACETransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Wan VACE Transformer 3D."""


class TestWanVACETransformer3DTraining(WanVACETransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Wan VACE Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"WanVACETransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestWanVACETransformer3DAttention(WanVACETransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Wan VACE Transformer 3D."""


class TestWanVACETransformer3DCompile(WanVACETransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Wan VACE Transformer 3D."""

    def test_torch_compile_repeated_blocks(self):
        # WanVACE has two block types (WanTransformerBlock and WanVACETransformerBlock),
        # so we need recompile_limit=2 instead of the default 1.
        super().test_torch_compile_repeated_blocks(recompile_limit=2)


class TestWanVACETransformer3DBitsAndBytes(WanVACETransformer3DTesterConfig, BitsAndBytesTesterMixin):
    """BitsAndBytes quantization tests for Wan VACE Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.float16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan VACE model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 16, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "control_hidden_states": randn_tensor(
                (1, 96, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanVACETransformer3DTorchAo(WanVACETransformer3DTesterConfig, TorchAoTesterMixin):
    """TorchAO quantization tests for Wan VACE Transformer 3D."""

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the tiny Wan VACE model dimensions."""
        return {
            "hidden_states": randn_tensor(
                (1, 16, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "control_hidden_states": randn_tensor(
                (1, 96, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanVACETransformer3DGGUF(WanVACETransformer3DTesterConfig, GGUFTesterMixin):
    """GGUF quantization tests for Wan VACE Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/blob/main/Wan2.1_14B_VACE-Q3_K_S.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan VACE model dimensions.

        Wan 2.1 VACE: in_channels=16, text_dim=4096, vace_in_channels=96
        """
        return {
            "hidden_states": randn_tensor(
                (1, 16, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "control_hidden_states": randn_tensor(
                (1, 96, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }


class TestWanVACETransformer3DGGUFCompile(WanVACETransformer3DTesterConfig, GGUFCompileTesterMixin):
    """GGUF + compile tests for Wan VACE Transformer 3D."""

    @property
    def gguf_filename(self):
        return "https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF/blob/main/Wan2.1_14B_VACE-Q3_K_S.gguf"

    @property
    def torch_dtype(self):
        return torch.bfloat16

    def get_dummy_inputs(self):
        """Override to provide inputs matching the real Wan VACE model dimensions.

        Wan 2.1 VACE: in_channels=16, text_dim=4096, vace_in_channels=96
        """
        return {
            "hidden_states": randn_tensor(
                (1, 16, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "encoder_hidden_states": randn_tensor(
                (1, 512, 4096), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "control_hidden_states": randn_tensor(
                (1, 96, 2, 64, 64), generator=self.generator, device=torch_device, dtype=self.torch_dtype
            ),
            "timestep": torch.tensor([1.0]).to(torch_device, self.torch_dtype),
        }
