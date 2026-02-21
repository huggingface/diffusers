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

from diffusers.models.transformers.ace_step_transformer import AceStepDiTModel
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TrainingTesterMixin,
)


enable_full_determinism()


class AceStepDiTModelTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AceStepDiTModel

    @property
    def output_shape(self):
        # (seq_len, acoustic_dim)
        return (8, 8)

    @property
    def input_shape(self):
        return (8, 8)

    @property
    def model_split_percents(self):
        return [0.9]

    @property
    def main_input_name(self):
        return "hidden_states"

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self):
        return {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "in_channels": 24,
            "audio_acoustic_hidden_dim": 8,
            "patch_size": 2,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "rms_norm_eps": 1e-6,
            "use_sliding_window": False,
            "sliding_window": 16,
        }

    def get_dummy_inputs(self):
        batch_size = 1
        seq_len = 8
        acoustic_dim = 8
        hidden_size = 32
        encoder_seq_len = 10

        return {
            "hidden_states": randn_tensor(
                (batch_size, seq_len, acoustic_dim), generator=self.generator, device=torch_device
            ),
            "timestep": torch.tensor([0.5], device=torch_device).expand(batch_size),
            "timestep_r": torch.tensor([0.3], device=torch_device).expand(batch_size),
            "encoder_hidden_states": randn_tensor(
                (batch_size, encoder_seq_len, hidden_size), generator=self.generator, device=torch_device
            ),
            "context_latents": randn_tensor(
                (batch_size, seq_len, acoustic_dim * 2), generator=self.generator, device=torch_device
            ),
        }


class TestAceStepDiTModel(AceStepDiTModelTesterConfig, ModelTesterMixin):
    """Core model tests for AceStepDiTModel."""

    def _check_dtype_inference_output(self, output, output_loaded, dtype, atol=2e-2, rtol=0):
        """Increase tolerance for half-precision inference with tiny random models."""
        super()._check_dtype_inference_output(output, output_loaded, dtype, atol=atol, rtol=rtol)


class TestAceStepDiTModelMemory(AceStepDiTModelTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AceStepDiTModel."""

    pass


class TestAceStepDiTModelTraining(AceStepDiTModelTesterConfig, TrainingTesterMixin):
    """Training tests for AceStepDiTModel."""

    pass
