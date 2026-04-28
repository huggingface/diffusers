# Copyright 2025 The HuggingFace Team.
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

from diffusers import JoyImageEditTransformer3DModel
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


class JoyImageEditTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return JoyImageEditTransformer3DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 1, 4, 4)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (16, 1, 4, 4)

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def uses_custom_attn_processor(self) -> bool:
        return True

    @property
    def model_split_percents(self) -> list:
        return [0.7, 0.6, 0.6]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "patch_size": [1, 2, 2],
            "in_channels": 16,
            "hidden_size": 32,
            "num_attention_heads": 2,
            "text_dim": 16,
            "num_layers": 2,
            "rope_dim_list": [4, 6, 6],
            "theta": 256,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        hidden_states = randn_tensor((batch_size, 16, 1, 4, 4), generator=self.generator, device=torch_device)
        encoder_hidden_states = randn_tensor((batch_size, 12, 16), generator=self.generator, device=torch_device)
        timestep = torch.tensor([1.0]).to(torch_device).expand(batch_size)
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
        }


class TestJoyImageEditTransformer(JoyImageEditTransformerTesterConfig, ModelTesterMixin):
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestJoyImageEditTransformerMemory(JoyImageEditTransformerTesterConfig, MemoryTesterMixin):
    pass


class TestJoyImageEditTransformerTraining(JoyImageEditTransformerTesterConfig, TrainingTesterMixin):
    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"JoyImageEditTransformer3DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestJoyImageEditTransformerAttention(JoyImageEditTransformerTesterConfig, AttentionTesterMixin):
    pass


class TestJoyImageEditTransformerCompile(JoyImageEditTransformerTesterConfig, TorchCompileTesterMixin):
    pass
