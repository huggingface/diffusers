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

"""
Unit tests for AutoencoderSAME.

The generic model test mixins (`ModelTesterMixin`, `TrainingTesterMixin`, `MemoryTesterMixin`) cover
save/load, determinism, dtype casting, and memory optimizations via a shared `forward()`. This file also
covers what those mixins don't reach: the `encode()` / `decode()` methods' own `return_dict` handling, and
`downsampling_ratio` arithmetic (including the production SAME-S preset).
"""

import unittest

import pytest
import torch

from diffusers import AutoencoderSAME
from diffusers.utils.torch_utils import randn_tensor

from ...testing_utils import require_torch, torch_device
from ..testing_utils import BaseModelTesterConfig, MemoryTesterMixin, ModelTesterMixin, TrainingTesterMixin


# ──────────────────────────────────────────────────────────────────────────────
# Tiny config used for fast unit testing (not the real SAME-S / SAME-L sizes)
# ──────────────────────────────────────────────────────────────────────────────

TINY_CFG = {
    "audio_channels": 2,
    "patch_size": 4,
    "encoder_channels": 16,
    "encoder_c_mults": [1, 2],
    "encoder_strides": [2, 2],
    "encoder_transformer_depths": [1, 1],
    "latent_dim": 8,
    "use_differential_attention": False,  # saves memory in tests
    "dim_heads": 8,
    "ff_mult": 2,
    "sampling_rate": 44100,
}

# Matches SAME-S architecture (single TRB, stride=16) with reduced depth for speed.
SAME_S_SMALL_CFG = {
    "audio_channels": 2,
    "patch_size": 256,
    "encoder_channels": 128,
    "encoder_c_mults": [6],
    "encoder_strides": [16],
    "encoder_transformer_depths": [2],  # 6 in production; reduced for test speed
    "latent_dim": 256,
    "use_differential_attention": False,
    "dim_heads": 64,
    "ff_mult": 3,
    "sampling_rate": 44100,
}

# 128 audio samples, TINY_CFG's downsampling_ratio = patch_size(4) * strides(2*2) = 16 → evenly divides 128,
# so decode(encode(x)) reproduces x's exact shape (no padding-derived length mismatch).
BATCH_SIZE = 2
T_AUDIO = 128


class AutoencoderSAMETesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderSAME

    @property
    def main_input_name(self) -> str:
        return "sample"

    @property
    def output_shape(self) -> tuple:
        return (TINY_CFG["audio_channels"], T_AUDIO)

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_init_dict(self) -> dict:
        return dict(TINY_CFG)

    def get_dummy_inputs(self) -> dict:
        audio = randn_tensor(
            (BATCH_SIZE, TINY_CFG["audio_channels"], T_AUDIO), generator=self.generator, device=torch_device
        )
        return {"sample": audio}


class TestAutoencoderSAME(AutoencoderSAMETesterConfig, ModelTesterMixin):
    pass


class TestAutoencoderSAMETraining(AutoencoderSAMETesterConfig, TrainingTesterMixin):
    """Training tests for AutoencoderSAME."""


class TestAutoencoderSAMEMemory(AutoencoderSAMETesterConfig, MemoryTesterMixin):
    """Memory optimization tests for AutoencoderSAME."""

    @pytest.mark.skip(
        "Test not supported because of 'weight_norm_fwd_first_dim_kernel' not implemented for 'Float8_e4m3fn'"
    )
    def test_layerwise_casting_training(self):
        super().test_layerwise_casting_training()

    @pytest.mark.skip(
        "The TRB `mapping` convs of AutoencoderSAME are wrapped with torch.nn.utils.weight_norm. This causes the "
        "hook's pre_forward to not cast the module weights to compute_dtype."
    )
    def test_layerwise_casting_memory(self):
        super().test_layerwise_casting_memory()


@require_torch
class TestAutoencoderSAMEBehavior(unittest.TestCase):
    """Coverage beyond the generic mixins: `encode()`/`decode()`'s own `return_dict` handling,
    `downsampling_ratio` arithmetic, and the SAME-S production-shape preset."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = AutoencoderSAME(**TINY_CFG).eval()
        self.audio = torch.randn(BATCH_SIZE, TINY_CFG["audio_channels"], T_AUDIO)

    def test_downsampling_ratio(self):
        expected = TINY_CFG["patch_size"]
        for s in TINY_CFG["encoder_strides"]:
            expected *= s
        self.assertEqual(self.model.downsampling_ratio, expected)

    def test_roundtrip_shape_consistency(self):
        latents = self.model.encode(self.audio).latents
        decoded = self.model.decode(latents).sample
        re_encoded = self.model.encode(decoded).latents
        self.assertEqual(latents.shape, re_encoded.shape)

    def test_encode_no_dict(self):
        out_tuple = self.model.encode(self.audio, return_dict=False)
        self.assertIsInstance(out_tuple, tuple)
        self.assertEqual(len(out_tuple), 1)
        self.assertIsInstance(out_tuple[0], torch.Tensor)

    def test_decode_no_dict(self):
        latents = self.model.encode(self.audio).latents
        out_tuple = self.model.decode(latents, return_dict=False)
        self.assertIsInstance(out_tuple, tuple)
        self.assertEqual(len(out_tuple), 1)

    def test_same_s_preset_downsampling_ratio(self):
        """Model with SAME-S production defaults should give 4096x ratio."""
        model = AutoencoderSAME(**SAME_S_SMALL_CFG)
        # patch_size=256, strides=[16] → 256×16 = 4096
        self.assertEqual(model.downsampling_ratio, 4096)


if __name__ == "__main__":
    unittest.main()
