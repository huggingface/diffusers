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

import gc
import unittest

import pytest
import torch
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_rae import (
    _ENCODER_TYPES,
    AutoencoderRAE,
    Dinov2Encoder,
    MAEEncoder,
    Siglip2Encoder,
)

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    slow,
    torch_device,
)
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


class TinyTestEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int = 16, patch_size: int = 8, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), kernel_size=self.patch_size, stride=self.patch_size)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return tokens.repeat(1, 1, self.hidden_size)


_ENCODER_TYPES["tiny_test"] = TinyTestEncoder


class AutoencoderRAETesterConfig(BaseModelTesterConfig):
    @property
    def model_class(self):
        return AutoencoderRAE

    @property
    def output_shape(self):
        return (3, 16, 16)

    def get_init_dict(self):
        return {
            "encoder_type": "tiny_test",
            "encoder_hidden_size": 16,
            "encoder_patch_size": 8,
            "encoder_input_size": 32,
            "patch_size": 4,
            "image_size": 16,
            "decoder_hidden_size": 32,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 4,
            "decoder_intermediate_size": 64,
            "num_channels": 3,
            "encoder_norm_mean": [0.5, 0.5, 0.5],
            "encoder_norm_std": [0.5, 0.5, 0.5],
            "noise_tau": 0.0,
            "reshape_to_2d": True,
            "scaling_factor": 1.0,
        }

    @property
    def generator(self):
        return torch.Generator("cpu").manual_seed(0)

    def get_dummy_inputs(self):
        return {"sample": torch.randn(2, 3, 32, 32, generator=self.generator, device="cpu").to(torch_device)}

    # Bridge for AutoencoderTesterMixin which still uses the old interface
    def prepare_init_args_and_inputs_for_common(self):
        return self.get_init_dict(), self.get_dummy_inputs()

    def _make_model(self, **overrides) -> AutoencoderRAE:
        config = self.get_init_dict()
        config.update(overrides)
        return AutoencoderRAE(**config).to(torch_device)


class TestAutoEncoderRAE(AutoencoderRAETesterConfig, ModelTesterMixin):
    """Core model tests for AutoencoderRAE."""

    @pytest.mark.skip(reason="AutoencoderRAE does not support gradient checkpointing yet")
    def test_from_save_pretrained_dynamo(self): ...

    def test_fast_encode_decode_and_forward_shapes(self):
        model = self._make_model().eval()
        x = torch.rand(2, 3, 32, 32, device=torch_device)

        with torch.no_grad():
            z = model.encode(x).latent
            decoded = model.decode(z).sample
            recon = model(x).sample

        assert z.shape == (2, 16, 4, 4)
        assert decoded.shape == (2, 3, 16, 16)
        assert recon.shape == (2, 3, 16, 16)
        assert torch.isfinite(recon).all().item()

    def test_fast_scaling_factor_encode_and_decode_consistency(self):
        torch.manual_seed(0)
        model_base = self._make_model(scaling_factor=1.0).eval()
        torch.manual_seed(0)
        model_scaled = self._make_model(scaling_factor=2.0).eval()

        x = torch.rand(2, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            z_base = model_base.encode(x).latent
            z_scaled = model_scaled.encode(x).latent
            recon_base = model_base.decode(z_base).sample
            recon_scaled = model_scaled.decode(z_scaled).sample

        assert torch.allclose(z_scaled, z_base * 2.0, atol=1e-5, rtol=1e-4)
        assert torch.allclose(recon_scaled, recon_base, atol=1e-5, rtol=1e-4)

    def test_fast_latents_normalization_matches_formula(self):
        latents_mean = torch.full((1, 16, 1, 1), 0.25, dtype=torch.float32)
        latents_std = torch.full((1, 16, 1, 1), 2.0, dtype=torch.float32)

        model_raw = self._make_model().eval()
        model_norm = self._make_model(latents_mean=latents_mean, latents_std=latents_std).eval()
        x = torch.rand(1, 3, 32, 32, device=torch_device)

        with torch.no_grad():
            z_raw = model_raw.encode(x).latent
            z_norm = model_norm.encode(x).latent

        expected = (z_raw - latents_mean.to(z_raw.device, z_raw.dtype)) / (
            latents_std.to(z_raw.device, z_raw.dtype) + 1e-5
        )
        assert torch.allclose(z_norm, expected, atol=1e-5, rtol=1e-4)

    def test_fast_slicing_matches_non_slicing(self):
        model = self._make_model().eval()
        x = torch.rand(3, 3, 32, 32, device=torch_device)

        with torch.no_grad():
            model.use_slicing = False
            z_no_slice = model.encode(x).latent
            out_no_slice = model.decode(z_no_slice).sample

            model.use_slicing = True
            z_slice = model.encode(x).latent
            out_slice = model.decode(z_slice).sample

        assert torch.allclose(z_slice, z_no_slice, atol=1e-6, rtol=1e-5)
        assert torch.allclose(out_slice, out_no_slice, atol=1e-6, rtol=1e-5)

    def test_fast_noise_tau_applies_only_in_train(self):
        model = self._make_model(noise_tau=0.5).to(torch_device)
        x = torch.rand(2, 3, 32, 32, device=torch_device)

        model.train()
        torch.manual_seed(0)
        z_train_1 = model.encode(x).latent
        torch.manual_seed(1)
        z_train_2 = model.encode(x).latent

        model.eval()
        torch.manual_seed(0)
        z_eval_1 = model.encode(x).latent
        torch.manual_seed(1)
        z_eval_2 = model.encode(x).latent

        assert z_train_1.shape == z_eval_1.shape
        assert not torch.allclose(z_train_1, z_train_2)
        assert torch.allclose(z_eval_1, z_eval_2, atol=1e-6, rtol=1e-5)


class TestAutoEncoderRAESlicingTiling(AutoencoderRAETesterConfig, AutoencoderTesterMixin):
    """Slicing and tiling tests for AutoencoderRAE."""


@slow
@pytest.mark.skip(reason="Not enough model usage to justify slow tests yet.")
class AutoencoderRAEEncoderIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_dinov2_encoder_forward_shape(self):
        encoder = Dinov2Encoder().to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 256
        assert y.shape[2] == encoder.model.config.hidden_size

    def test_siglip2_encoder_forward_shape(self):
        encoder = Siglip2Encoder().to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196
        assert y.shape[2] == encoder.model.config.hidden_size

    def test_mae_encoder_forward_shape(self):
        encoder = MAEEncoder().to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196
        assert y.shape[2] == encoder.model.config.hidden_size


@slow
@pytest.mark.skip(reason="Not enough model usage to justify slow tests yet.")
class AutoencoderRAEIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_autoencoder_rae_from_pretrained_dinov2(self):
        model = AutoencoderRAE.from_pretrained("nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08").to(torch_device)
        model.eval()

        x = torch.rand(1, 3, 224, 224, device=torch_device)

        with torch.no_grad():
            latents = model.encode(x).latent
            assert latents.ndim == 4
            assert latents.shape[0] == 1

            decoded = model.decode(latents).sample
            assert decoded.shape[0] == 1
            assert decoded.shape[1] == 3

            recon = model(x).sample
            assert torch.isfinite(recon).all().item()
