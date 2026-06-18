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

from diffusers import SanaVideoTransformer3DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class SanaVideoTransformer3DTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return SanaVideoTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (16, 2, 16, 16)

    @property
    def output_shape(self) -> tuple:
        return (16, 2, 16, 16)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 2,
            "attention_head_dim": 12,
            "num_layers": 2,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 12,
            "cross_attention_dim": 24,
            "caption_channels": 16,
            "mlp_ratio": 2.5,
            "dropout": 0.0,
            "attention_bias": False,
            "sample_size": 8,
            "patch_size": (1, 2, 2),
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "qk_norm": "rms_norm_across_heads",
            "rope_max_seq_len": 32,
        }

    def get_dummy_inputs(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        num_channels = 16
        num_frames = 2
        height = width = 16
        text_encoder_embedding_dim = 16
        sequence_length = 12

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, text_encoder_embedding_dim),
                generator=self.generator,
                device=torch_device,
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
        }


class TestSanaVideoTransformer3D(SanaVideoTransformer3DTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestSanaVideoTransformer3DMemory(SanaVideoTransformer3DTesterConfig, MemoryTesterMixin):
    pass


class TestSanaVideoTransformer3DAttention(SanaVideoTransformer3DTesterConfig, AttentionTesterMixin):
    pass


class TestSanaVideoTransformer3DTraining(SanaVideoTransformer3DTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"SanaVideoTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
