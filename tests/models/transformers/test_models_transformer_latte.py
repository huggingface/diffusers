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

import pytest
import torch

from diffusers import LatteTransformer3DModel
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


class LatteTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return LatteTransformer3DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def input_shape(self) -> tuple:
        return (4, 1, 8, 8)

    @property
    def output_shape(self) -> tuple:
        return (8, 1, 8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "sample_size": 8,
            "num_layers": 1,
            "patch_size": 2,
            "attention_head_dim": 4,
            "num_attention_heads": 2,
            "caption_channels": 8,
            "in_channels": 4,
            "cross_attention_dim": 8,
            "out_channels": 8,
            "attention_bias": True,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "norm_type": "ada_norm_single",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
        }

    def get_dummy_inputs(self, batch_size: int = 2) -> dict[str, torch.Tensor]:
        num_channels = 4
        num_frames = 1
        height = width = 8
        embedding_dim = 8
        sequence_length = 8

        return {
            "hidden_states": randn_tensor(
                (batch_size, num_channels, num_frames, height, width), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, sequence_length, embedding_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.randint(0, 1000, size=(batch_size,), generator=self.generator).to(torch_device),
            "enable_temporal_attentions": True,
        }


class TestLatteTransformer(LatteTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: fp16/bf16 require very high atol to pass, providing little signal.
        # Dtype preservation is already tested by test_from_save_pretrained_dtype and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestLatteTransformerMemory(LatteTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestLatteTransformerAttention(LatteTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestLatteTransformerTraining(LatteTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"LatteTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)
