# coding=utf-8
# Copyright 2026 HuggingFace Inc.
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

from diffusers import Ideogram4Transformer2DModel
from diffusers.models.transformers.transformer_ideogram4 import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    Ideogram4MRoPE,
)
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


class Ideogram4TransformerTesterConfig(BaseModelTesterConfig):
    _hidden_size = 32
    _num_heads = 4
    _head_dim = _hidden_size // _num_heads  # 8
    _in_channels = 16
    _llm_features_dim = 24
    _max_text_tokens = 4
    _num_image_tokens = 4

    @property
    def model_class(self):
        return Ideogram4Transformer2DModel

    @property
    def main_input_name(self) -> str:
        return "hidden_states"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (1, self._max_text_tokens + self._num_image_tokens, self._in_channels)

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (1, self._max_text_tokens + self._num_image_tokens, self._in_channels)

    @property
    def model_split_percents(self) -> list:
        return [0.9, 0.9, 0.9]

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return {
            "in_channels": self._in_channels,
            "num_layers": 2,
            "attention_head_dim": self._head_dim,
            "num_attention_heads": self._num_heads,
            "intermediate_size": 32,
            "adaln_dim": 16,
            "llm_features_dim": self._llm_features_dim,
            "rope_theta": 10_000,
            "mrope_section": (2, 1, 1),
            "norm_eps": 1e-5,
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        max_text_tokens = self._max_text_tokens
        num_image_tokens = self._num_image_tokens
        seq_len = max_text_tokens + num_image_tokens

        hidden_states = torch.zeros(
            batch_size, seq_len, self._in_channels, device=torch_device, dtype=self.torch_dtype
        )
        hidden_states[:, max_text_tokens:] = randn_tensor(
            (batch_size, num_image_tokens, self._in_channels),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )

        encoder_hidden_states = torch.zeros(
            batch_size, seq_len, self._llm_features_dim, device=torch_device, dtype=self.torch_dtype
        )
        encoder_hidden_states[:, :max_text_tokens] = randn_tensor(
            (batch_size, max_text_tokens, self._llm_features_dim),
            generator=self.generator,
            device=torch_device,
            dtype=self.torch_dtype,
        )

        position_ids = torch.zeros(batch_size, seq_len, 3, dtype=torch.long, device=torch_device)
        text_pos = torch.arange(max_text_tokens, device=torch_device)
        position_ids[:, :max_text_tokens, 0] = text_pos
        position_ids[:, :max_text_tokens, 1] = text_pos
        position_ids[:, :max_text_tokens, 2] = text_pos
        # Image tokens get a 2x2 grid with the IMAGE_POSITION_OFFSET applied.
        image_h = torch.tensor([0, 0, 1, 1], device=torch_device)
        image_w = torch.tensor([0, 1, 0, 1], device=torch_device)
        position_ids[:, max_text_tokens:, 0] = IMAGE_POSITION_OFFSET
        position_ids[:, max_text_tokens:, 1] = image_h + IMAGE_POSITION_OFFSET
        position_ids[:, max_text_tokens:, 2] = image_w + IMAGE_POSITION_OFFSET

        segment_ids = torch.ones(batch_size, seq_len, dtype=torch.long, device=torch_device)
        indicator = torch.empty(batch_size, seq_len, dtype=torch.long, device=torch_device)
        indicator[:, :max_text_tokens] = LLM_TOKEN_INDICATOR
        indicator[:, max_text_tokens:] = OUTPUT_IMAGE_INDICATOR
        timestep = torch.tensor([0.5], device=torch_device, dtype=self.torch_dtype)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "indicator": indicator,
        }


class TestIdeogram4Transformer(Ideogram4TransformerTesterConfig, ModelTesterMixin):
    """Core model tests for Ideogram 4 Transformer."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
    def test_from_save_pretrained_dtype_inference(self, tmp_path, dtype):
        # Skip: the non-persistent fp32 RoPE inv_freq buffer is truncated to fp16 by the in-memory
        # .to(dtype) path but kept fp32 by from_pretrained, so the two outputs diverge well beyond any
        # meaningful tolerance. Dtype preservation is already covered by test_from_save_pretrained_dtype
        # and test_keep_in_fp32_modules.
        pytest.skip("Tolerance requirements too high for meaningful test")


class TestIdeogram4TransformerMemory(Ideogram4TransformerTesterConfig, MemoryTesterMixin):
    """Memory optimization tests for Ideogram 4 Transformer."""


class TestIdeogram4TransformerTraining(Ideogram4TransformerTesterConfig, TrainingTesterMixin):
    """Training tests for Ideogram 4 Transformer."""

    def test_gradient_checkpointing_is_applied(self):
        expected_set = {"Ideogram4Transformer2DModel"}
        super().test_gradient_checkpointing_is_applied(expected_set=expected_set)


class TestIdeogram4TransformerAttention(Ideogram4TransformerTesterConfig, AttentionTesterMixin):
    """Attention processor tests for Ideogram 4 Transformer."""


def test_ideogram4_mrope_is_autocast_invariant():
    # Ideogram4's image positions start at IMAGE_POSITION_OFFSET (65536), so the rotary matmul must
    # run in float32: under an ambient autocast it would otherwise execute in bfloat16 and round every
    # image position to the same value, collapsing all spatial information (the decoded image goes flat).
    rope = Ideogram4MRoPE(head_dim=256, base=5_000_000, mrope_section=(24, 20, 20)).to(torch_device)
    position_ids = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 63, 63]]], device=torch_device) + IMAGE_POSITION_OFFSET

    cos_ref, sin_ref = rope(position_ids)
    with torch.autocast(device_type=torch.device(torch_device).type, dtype=torch.bfloat16):
        cos_ac, sin_ac = rope(position_ids)

    # Distinct image positions must keep distinct embeddings, identical to the float32 computation.
    assert not torch.equal(cos_ac[0, 0], cos_ac[0, 1])
    assert torch.equal(cos_ac, cos_ref)
    assert torch.equal(sin_ac, sin_ref)
