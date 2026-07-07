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

Covers:
  - I/O tensor shapes for encode and decode
  - Round-trip reconstruct (decode(encode(x)) ≈ x shape)
  - downsampling_ratio matches config arithmetic
"""

import unittest

import torch

from diffusers import AutoencoderSAME

from ...testing_utils import require_torch, torch_device


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
    "encoder_chunk_size": 4,
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
    "encoder_chunk_size": 32,
    "ff_mult": 3,
    "sampling_rate": 44100,
}


def _make_model(cfg: dict, device: str = "cpu") -> AutoencoderSAME:
    model = AutoencoderSAME(**cfg)
    model.to(device).eval()
    return model


def _make_audio(batch: int, channels: int, n_samples: int, device: str = "cpu") -> torch.Tensor:
    return torch.randn(batch, channels, n_samples, device=device)


@require_torch
class TestAutoencoderSAMETinyConfig(unittest.TestCase):
    """Fast tests using the tiny config — runs on CPU in seconds."""

    def setUp(self):
        torch.manual_seed(0)
        self.model = _make_model(TINY_CFG)
        self.B = 2
        self.C = TINY_CFG["audio_channels"]
        # 128 audio samples → 128 / (patch×strides) = 128 / 16 = 8 latent frames
        self.T = 128
        self.audio = _make_audio(self.B, self.C, self.T)

    # ------------------------------------------------------------------
    def test_downsampling_ratio(self):
        expected = TINY_CFG["patch_size"]
        for s in TINY_CFG["encoder_strides"]:
            expected *= s
        self.assertEqual(self.model.downsampling_ratio, expected)

    # ------------------------------------------------------------------
    def test_encode_output_shape(self):
        out = self.model.encode(self.audio)
        latents = out.latents
        T_lat = self.T // self.model.downsampling_ratio
        self.assertEqual(latents.shape, (self.B, TINY_CFG["latent_dim"], T_lat))

    # ------------------------------------------------------------------
    def test_decode_output_shape(self):
        latents = self.model.encode(self.audio).latents
        decoded = self.model.decode(latents).sample
        # Decoded length may differ from original because of padding in encode
        self.assertEqual(decoded.shape[0], self.B)
        self.assertEqual(decoded.shape[1], self.C)
        self.assertGreaterEqual(decoded.shape[2], self.T)

    # ------------------------------------------------------------------
    def test_roundtrip_shape_consistency(self):
        latents = self.model.encode(self.audio).latents
        decoded = self.model.decode(latents).sample
        re_encoded = self.model.encode(decoded).latents
        self.assertEqual(latents.shape, re_encoded.shape)

    # ------------------------------------------------------------------
    def test_encode_no_dict(self):
        out_tuple = self.model.encode(self.audio, return_dict=False)
        self.assertIsInstance(out_tuple, tuple)
        self.assertEqual(len(out_tuple), 1)
        self.assertIsInstance(out_tuple[0], torch.Tensor)

    # ------------------------------------------------------------------
    def test_decode_no_dict(self):
        latents = self.model.encode(self.audio).latents
        out_tuple = self.model.decode(latents, return_dict=False)
        self.assertIsInstance(out_tuple, tuple)
        self.assertEqual(len(out_tuple), 1)

    # ------------------------------------------------------------------
    def test_different_batch_sizes(self):
        for b in (1, 3, 4):
            audio = _make_audio(b, self.C, self.T)
            lat = self.model.encode(audio).latents
            self.assertEqual(lat.shape[0], b)

    # ------------------------------------------------------------------
    def test_config_serialisation_round_trip(self):
        cfg = self.model.config
        model2 = AutoencoderSAME(**{k: v for k, v in cfg.items() if not k.startswith("_")})
        self.assertEqual(model2.downsampling_ratio, self.model.downsampling_ratio)
        self.assertEqual(model2.latent_dim, self.model.latent_dim)

    # ------------------------------------------------------------------
    def test_forward_shape(self):
        out = self.model(self.audio)
        self.assertIsInstance(out.sample, torch.Tensor)
        self.assertEqual(out.sample.shape[:2], (self.B, self.C))

    # ------------------------------------------------------------------
    def test_forward_no_dict(self):
        out = self.model(self.audio, return_dict=False)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 1)

    # ------------------------------------------------------------------
    def test_same_s_preset_downsampling_ratio(self):
        """Model with SAME-S production defaults should give 4096× ratio."""
        model = _make_model(SAME_S_SMALL_CFG)
        # patch_size=256, strides=[16] → 256×16 = 4096
        self.assertEqual(model.downsampling_ratio, 4096)

    # ------------------------------------------------------------------
    def test_training_mode(self):
        self.model.train()
        audio = _make_audio(1, self.C, self.T)
        lat = self.model.encode(audio).latents
        dec = self.model.decode(lat).sample
        self.assertEqual(lat.ndim, 3)
        self.assertEqual(dec.ndim, 3)


@require_torch
class TestAutoencoderSAMEOnDevice(unittest.TestCase):
    """Tests that the tiny model runs on whatever device pytest selects."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch_device
        self.model = _make_model(TINY_CFG, device=str(self.device))

    def test_encode_on_device(self):
        audio = _make_audio(1, 2, 64, device=str(self.device))
        lat = self.model.encode(audio).latents
        self.assertEqual(lat.device.type, torch.device(self.device).type)

    def test_decode_on_device(self):
        audio = _make_audio(1, 2, 64, device=str(self.device))
        lat = self.model.encode(audio).latents
        dec = self.model.decode(lat).sample
        self.assertEqual(dec.device.type, torch.device(self.device).type)


if __name__ == "__main__":
    unittest.main()
