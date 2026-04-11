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

from diffusers import LongCatAudioDiTTransformer
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import enable_full_determinism, torch_device
from ..testing_utils import (
    AttentionTesterMixin,
    BaseModelTesterConfig,
    MemoryTesterMixin,
    ModelTesterMixin,
    TorchCompileTesterMixin,
)


enable_full_determinism()


class LongCatAudioDiTTransformerTesterConfig(BaseModelTesterConfig):
    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def model_class(self):
        return LongCatAudioDiTTransformer

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (16, 8)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | bool | float | str]:
        return {
            "dit_dim": 64,
            "dit_depth": 2,
            "dit_heads": 4,
            "dit_text_dim": 32,
            "latent_dim": 8,
            "text_conv": False,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        sequence_length = 16
        encoder_sequence_length = 10
        latent_dim = 8
        text_dim = 32

        return {
            "hidden_states": randn_tensor(
                (batch_size, sequence_length, latent_dim), generator=self.generator, device=torch_device
            ),
            "encoder_hidden_states": randn_tensor(
                (batch_size, encoder_sequence_length, text_dim), generator=self.generator, device=torch_device
            ),
            "encoder_attention_mask": torch.ones(
                batch_size, encoder_sequence_length, dtype=torch.bool, device=torch_device
            ),
            "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.bool, device=torch_device),
            "timestep": torch.ones(batch_size, device=torch_device),
        }


class TestLongCatAudioDiTTransformer(LongCatAudioDiTTransformerTesterConfig, ModelTesterMixin):
    pass


class TestLongCatAudioDiTTransformerMemory(LongCatAudioDiTTransformerTesterConfig, MemoryTesterMixin):
    def test_layerwise_casting_memory(self):
        pytest.skip(
            "LongCatAudioDiTTransformer tiny test config does not provide stable layerwise casting peak memory "
            "coverage."
        )


class TestLongCatAudioDiTTransformerCompile(LongCatAudioDiTTransformerTesterConfig, TorchCompileTesterMixin):
    pass


class TestLongCatAudioDiTTransformerAttention(LongCatAudioDiTTransformerTesterConfig, AttentionTesterMixin):
    pass


def test_longcat_audio_attention_uses_standard_self_attn_kwargs():
    from diffusers.models.transformers.transformer_longcat_audio_dit import AudioDiTAttention

    attn = AudioDiTAttention(q_dim=4, kv_dim=None, heads=1, dim_head=4, dropout=0.0, bias=False)

    eye = torch.eye(4)
    with torch.no_grad():
        attn.to_q.weight.copy_(eye)
        attn.to_k.weight.copy_(eye)
        attn.to_v.weight.copy_(eye)
        attn.to_out[0].weight.copy_(eye)

    hidden_states = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.5]]])
    attention_mask = torch.tensor([[True, False]])

    output = attn(hidden_states=hidden_states, attention_mask=attention_mask)

    assert torch.allclose(output[:, 1], torch.zeros_like(output[:, 1]))
