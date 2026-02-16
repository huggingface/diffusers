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

import torch
import torch.nn.functional as F

from diffusers.models.autoencoders.autoencoder_rae import (
    AutoencoderRAE,
    Dinov2Encoder,
    MAEEncoder,
    Siglip2Encoder,
    register_encoder,
)

from ...testing_utils import backend_empty_cache, enable_full_determinism, slow, torch_device


enable_full_determinism()


DINO_MODEL_ID = "facebook/dinov2-with-registers-base"
SIGLIP2_MODEL_ID = "google/siglip2-base-patch16-256"
MAE_MODEL_ID = "facebook/vit-mae-base"


@register_encoder(name="tiny_test")
class TinyTestEncoder(torch.nn.Module):
    def __init__(self, encoder_name_or_path: str = "unused"):
        super().__init__()
        self.patch_size = 8
        self.hidden_size = 16

    def forward(self, images: torch.Tensor, requires_grad: bool = False) -> torch.Tensor:
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), kernel_size=self.patch_size, stride=self.patch_size)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return tokens.repeat(1, 1, self.hidden_size)


class AutoencoderRAETests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def _make_model(self, **overrides) -> AutoencoderRAE:
        config = {
            "encoder_cls": "tiny_test",
            "encoder_name_or_path": "unused",
            "encoder_input_size": 32,
            "patch_size": 4,
            "image_size": 16,
            "decoder_hidden_size": 32,
            "decoder_num_hidden_layers": 1,
            "decoder_num_attention_heads": 4,
            "decoder_intermediate_size": 64,
            "num_channels": 3,
            "noise_tau": 0.0,
            "reshape_to_2d": True,
            "scaling_factor": 1.0,
        }
        config.update(overrides)
        return AutoencoderRAE(**config).to(torch_device)

    def test_fast_encode_decode_and_forward_shapes(self):
        model = self._make_model().eval()
        x = torch.rand(2, 3, 32, 32, device=torch_device)

        with torch.no_grad():
            z = model.encode(x).latent
            decoded = model.decode(z).sample
            recon = model(x).sample

        self.assertEqual(z.shape, (2, 16, 4, 4))
        self.assertEqual(decoded.shape, (2, 3, 16, 16))
        self.assertEqual(recon.shape, (2, 3, 16, 16))
        self.assertTrue(torch.isfinite(recon).all().item())

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

        self.assertTrue(torch.allclose(z_scaled, z_base * 2.0, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(recon_scaled, recon_base, atol=1e-5, rtol=1e-4))

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
        self.assertTrue(torch.allclose(z_norm, expected, atol=1e-5, rtol=1e-4))

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

        self.assertTrue(torch.allclose(z_slice, z_no_slice, atol=1e-6, rtol=1e-5))
        self.assertTrue(torch.allclose(out_slice, out_no_slice, atol=1e-6, rtol=1e-5))

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

        self.assertEqual(z_train_1.shape, z_eval_1.shape)
        self.assertFalse(torch.allclose(z_train_1, z_train_2))
        self.assertTrue(torch.allclose(z_eval_1, z_eval_2, atol=1e-6, rtol=1e-5))

    def test_fast_forward_return_loss_reconstruction_only(self):
        model = self._make_model(use_encoder_loss=False).train()
        x = torch.rand(2, 3, 16, 16, device=torch_device)

        output = model(x, return_loss=True)

        self.assertEqual(output.sample.shape, (2, 3, 16, 16))
        self.assertTrue(torch.isfinite(output.loss).all().item())
        self.assertTrue(torch.isfinite(output.reconstruction_loss).all().item())
        self.assertTrue(torch.isfinite(output.encoder_loss).all().item())
        self.assertEqual(output.encoder_loss.item(), 0.0)
        self.assertTrue(torch.allclose(output.loss, output.reconstruction_loss))

    def test_fast_forward_return_loss_with_encoder_loss(self):
        model = self._make_model(use_encoder_loss=True).train()
        x = torch.rand(2, 3, 16, 16, device=torch_device)

        output = model(x, return_loss=True, encoder_loss_weight=0.5, reconstruction_loss_type="mse")

        self.assertEqual(output.sample.shape, (2, 3, 16, 16))
        self.assertTrue(torch.isfinite(output.loss).all().item())
        self.assertTrue(torch.isfinite(output.reconstruction_loss).all().item())
        self.assertTrue(torch.isfinite(output.encoder_loss).all().item())
        self.assertGreaterEqual(output.encoder_loss.item(), 0.0)
        self.assertGreaterEqual(output.loss.item(), output.reconstruction_loss.item())


@slow
class AutoencoderRAEEncoderIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_dinov2_encoder_forward_shape(self):
        dino_path = DINO_MODEL_ID

        encoder = Dinov2Encoder(encoder_name_or_path=dino_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 256
        assert y.shape[2] == encoder.hidden_size

    def test_siglip2_encoder_forward_shape(self):
        siglip2_path = SIGLIP2_MODEL_ID

        encoder = Siglip2Encoder(encoder_name_or_path=siglip2_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196
        assert y.shape[2] == encoder.hidden_size

    def test_mae_encoder_forward_shape(self):
        mae_path = MAE_MODEL_ID

        encoder = MAEEncoder(encoder_name_or_path=mae_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196
        assert y.shape[2] == encoder.hidden_size


@slow
class AutoencoderRAEIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_autoencoder_rae_encode_decode_forward_shapes_dinov2(self):
        # This is a shape & numerical-sanity test. The decoder is randomly initialized unless you load trained weights.
        dino_path = DINO_MODEL_ID

        encoder_input_size = 224
        decoder_patch_size = 16
        # dinov2 patch=14 -> (224/14)^2 = 256 tokens -> decoder output 256 for patch 16
        image_size = 256

        model = AutoencoderRAE(
            encoder_cls="dinov2",
            encoder_name_or_path=dino_path,
            image_size=image_size,
            encoder_input_size=encoder_input_size,
            patch_size=decoder_patch_size,
            # keep the decoder lightweight for test runtime
            decoder_hidden_size=128,
            decoder_num_hidden_layers=1,
            decoder_num_attention_heads=4,
            decoder_intermediate_size=256,
        ).to(torch_device)
        model.eval()

        x = torch.rand(1, 3, encoder_input_size, encoder_input_size, device=torch_device)

        with torch.no_grad():
            latents = model.encode(x).latent
            assert latents.ndim == 4
            assert latents.shape[0] == 1

            decoded = model.decode(latents).sample
            assert decoded.shape == (1, 3, image_size, image_size)

            recon = model(x).sample
            assert recon.shape == (1, 3, image_size, image_size)
            assert torch.isfinite(recon).all().item()
