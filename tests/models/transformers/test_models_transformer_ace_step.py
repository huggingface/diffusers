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

import torch

from diffusers import AceStepTransformer1DModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin


enable_full_determinism()


class AceStepTransformer1DModelTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_class(self):
        return AceStepTransformer1DModel

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (8, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | float | bool]:
        return {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "in_channels": 24,  # audio_acoustic_hidden_dim * 3 (hidden + context_latents)
            "audio_acoustic_hidden_dim": 8,
            "patch_size": 2,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6,
            "sliding_window": 16,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 2
        seq_len = 8
        encoder_seq_len = 10
        acoustic_dim = 8
        hidden_size = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, seq_len, acoustic_dim), generator=self.generator, device=torch_device
            ),
            "timestep": randn_tensor((batch_size,), generator=self.generator, device=torch_device).abs(),
            "timestep_r": randn_tensor((batch_size,), generator=self.generator, device=torch_device).abs(),
            "encoder_hidden_states": randn_tensor(
                (batch_size, encoder_seq_len, hidden_size), generator=self.generator, device=torch_device
            ),
            "context_latents": randn_tensor(
                (batch_size, seq_len, acoustic_dim * 2), generator=self.generator, device=torch_device
            ),
        }


class TestAceStepTransformer1DModel(AceStepTransformer1DModelTesterConfig, ModelTesterMixin):
    pass
