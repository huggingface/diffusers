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
import torch.nn.functional as F

from diffusers import AutoencoderKLJoyVideoEdit
from diffusers.models.autoencoders.autoencoder_kl_joyvideoedit import JoyVideoEditAttentionBlock
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


class AutoencoderKLJoyVideoEditTesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderKLJoyVideoEdit

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (3, 5, 24, 24)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict[str, int | list[int]]:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "patch_size": 1,
            "latent_channels": 4,
            "layers_per_block": 1,
            "block_in_channels": (8, 8),
            "temporal_downsample": (True, False),
            "chunk_size": 4,
            "latents_mean": [0.0, 0.0, 0.0, 0.0],
            "latents_std": [1.0, 1.0, 1.0, 1.0],
        }

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        batch_size = 1
        num_channels = 3
        num_frames = 5  # temporal_compression_ratio * n + 1
        sizes = (24, 24)  # divisible by the spatial compression ratio (3 in this dummy config)
        image = randn_tensor(
            (batch_size, num_channels, num_frames, *sizes), generator=self.generator, device=torch_device
        )
        return {"sample": image}


class TestAutoencoderKLJoyVideoEditModel(AutoencoderKLJoyVideoEditTesterConfig, ModelTesterMixin):
    base_precision = 1e-2

    @pytest.mark.parametrize("chunk_size", [0, 1])
    def test_invalid_chunk_size_raises(self, chunk_size):
        init_dict = self.get_init_dict()
        init_dict["chunk_size"] = chunk_size

        with pytest.raises(ValueError, match="positive multiple of the temporal compression ratio"):
            self.model_class(**init_dict)

    def test_last_temporal_downsample_must_be_false(self):
        init_dict = self.get_init_dict()
        init_dict["temporal_downsample"] = (True, True)

        with pytest.raises(ValueError, match="last value must be `False`"):
            self.model_class(**init_dict)

    def test_temporal_compression_ratio_one(self):
        init_dict = self.get_init_dict()
        init_dict["temporal_downsample"] = (False, False)
        model = self.model_class(**init_dict).to(torch_device)
        sample = randn_tensor((1, 3, 1, 24, 24), generator=self.generator, device=torch_device)

        latents = model.encode(sample).latent_dist.mode()

        assert latents.shape[2] == 1

    def test_attention_processor_matches_sdpa(self):
        attention = JoyVideoEditAttentionBlock(8).to(torch_device)
        hidden_states = randn_tensor((2, 8, 3, 4, 4), generator=self.generator, device=torch_device)

        actual = attention(hidden_states)

        normed_hidden_states = attention.norm(hidden_states)
        batch_size, channels, num_frames, height, width = hidden_states.shape
        query = attention.q(normed_hidden_states)
        key = attention.k(normed_hidden_states)
        value = attention.v(normed_hidden_states)
        query = query.permute(0, 2, 3, 4, 1).reshape(batch_size * num_frames, 1, height * width, channels)
        key = key.permute(0, 2, 3, 4, 1).reshape(batch_size * num_frames, 1, height * width, channels)
        value = value.permute(0, 2, 3, 4, 1).reshape(batch_size * num_frames, 1, height * width, channels)
        expected = F.scaled_dot_product_attention(query, key, value)
        expected = expected.reshape(batch_size, num_frames, height, width, channels).permute(0, 4, 1, 2, 3)
        expected = hidden_states + attention.proj_out(expected)

        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


class TestAutoencoderKLJoyVideoEditMemory(AutoencoderKLJoyVideoEditTesterConfig, MemoryTesterMixin):
    pass


class TestAutoencoderKLJoyVideoEditCompile(AutoencoderKLJoyVideoEditTesterConfig, TorchCompileTesterMixin):
    pass


class TestAutoencoderKLJoyVideoEditAttention(AutoencoderKLJoyVideoEditTesterConfig, AttentionTesterMixin):
    pass
