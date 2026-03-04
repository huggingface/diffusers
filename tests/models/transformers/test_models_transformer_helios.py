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

from diffusers import HeliosTransformer3DModel
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


class HeliosTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return HeliosTransformer3DModel

    @property
    def pretrained_model_name_or_path(self):
        return "hf-internal-testing/tiny-helios-base-transformer"

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
            "rope_dim": (4, 4, 4),
            "has_multi_term_memory_patch": True,
            "guidance_cross_attn": True,
            "zero_history_timestep": True,
            "is_amplify_history": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 4
        num_frames = 2
        height = 16
        width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        hidden_states = randn_tensor(
            (batch_size, num_channels, num_frames, height, width),
            generator=self.generator,
            device=torch_device,
        )
        timestep = torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device)
        encoder_hidden_states = randn_tensor(
            (batch_size, sequence_length, text_encoder_embedding_dim),
            generator=self.generator,
            device=torch_device,
        )
        indices_hidden_states = torch.ones((batch_size, num_frames)).to(torch_device)
        indices_latents_history_short = torch.ones((batch_size, num_frames - 1)).to(torch_device)
        indices_latents_history_mid = torch.ones((batch_size, num_frames - 1)).to(torch_device)
        indices_latents_history_long = torch.ones((batch_size, (num_frames - 1) * 4)).to(torch_device)
        latents_history_short = randn_tensor(
            (batch_size, num_channels, num_frames - 1, height, width),
            generator=self.generator,
            device=torch_device,
        )
        latents_history_mid = randn_tensor(
            (batch_size, num_channels, num_frames - 1, height, width),
            generator=self.generator,
            device=torch_device,
        )
        latents_history_long = randn_tensor(
            (batch_size, num_channels, (num_frames - 1) * 4, height, width),
            generator=self.generator,
            device=torch_device,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "indices_hidden_states": indices_hidden_states,
            "indices_latents_history_short": indices_latents_history_short,
            "indices_latents_history_mid": indices_latents_history_mid,
            "indices_latents_history_long": indices_latents_history_long,
            "latents_history_short": latents_history_short,
            "latents_history_mid": latents_history_mid,
            "latents_history_long": latents_history_long,
        }


class TestHeliosTransformer3D(HeliosTransformer3DTesterConfig, ModelTesterMixin):
    """Core model tests for Helios Transformer 3D."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestHeliosTransformer3DMemory(HeliosTransformer3DTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Helios Transformer 3D."""


class TestHeliosTransformer3DTraining(HeliosTransformer3DTesterConfig, TrainingTesterMixin):
    """Training tests for Helios Transformer 3D."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"HeliosTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestHeliosTransformer3DAttention(HeliosTransformer3DTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Helios Transformer 3D."""


class TestHeliosTransformer3DCompile(HeliosTransformer3DTesterConfig, TorchCompileTesterMixin):
    """Torch compile tests for Helios Transformer 3D."""

    @pytest.mark.xfail(
        reason="Helios DiT does not compile when deterministic algorithms are used due to https://github.com/pytorch/pytorch/issues/170079"
    )
    def test_torch_compile_recompilation_and_graph_break(self):
        super().test_torch_compile_recompilation_and_graph_break()
