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

import pytest
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor

import diffusers.models.autoencoders.autoencoder_rae as _rae_module
from diffusers.models.autoencoders.autoencoder_rae import (
    _ENCODER_FORWARD_FNS,
    AutoencoderRAE,
    _build_encoder,
)
from diffusers.utils import load_image

from ...testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    slow,
    torch_all_close,
    torch_device,
)
from ..testing_utils import BaseModelTesterConfig, ModelTesterMixin
from .testing_utils import AutoencoderTesterMixin


enable_full_determinism()


# ---------------------------------------------------------------------------
# Tiny test encoder for fast unit tests (no transformers dependency)
# ---------------------------------------------------------------------------


class _TinyTestEncoderModule(torch.nn.Module):
    """Minimal encoder that mimics the patch-token interface without any HF model."""

    def __init__(self, hidden_size: int = 16, patch_size: int = 8, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pooled = F.avg_pool2d(images.mean(dim=1, keepdim=True), kernel_size=self.patch_size, stride=self.patch_size)
        tokens = pooled.flatten(2).transpose(1, 2).contiguous()
        return tokens.repeat(1, 1, self.hidden_size)


def _tiny_test_encoder_forward(model, images):
    return model(images)


def _build_tiny_test_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers):
    return _TinyTestEncoderModule(hidden_size=hidden_size, patch_size=patch_size)


# Monkey-patch the dispatch tables so "tiny_test" is recognised by AutoencoderRAE
_ENCODER_FORWARD_FNS["tiny_test"] = _tiny_test_encoder_forward
_original_build_encoder = _build_encoder


def _patched_build_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers):
    if encoder_type == "tiny_test":
        return _build_tiny_test_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers)
    return _original_build_encoder(encoder_type, hidden_size, patch_size, num_hidden_layers)


_rae_module._build_encoder = _patched_build_encoder


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------


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

    @pytest.mark.skip(reason="AutoencoderRAE does not support torch dynamo yet")
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
class AutoencoderRAEEncoderIntegrationTests:
    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_dinov2_encoder_forward_shape(self):
        encoder = _build_encoder("dinov2", hidden_size=768, patch_size=14, num_hidden_layers=12).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = _ENCODER_FORWARD_FNS["dinov2"](encoder, x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 256  # (224/14)^2 - 5 (CLS + 4 register) = 251?  Actually dinov2 has 256 patches
        assert y.shape[2] == 768

    def test_siglip2_encoder_forward_shape(self):
        encoder = _build_encoder("siglip2", hidden_size=768, patch_size=16, num_hidden_layers=12).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = _ENCODER_FORWARD_FNS["siglip2"](encoder, x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196  # (224/16)^2
        assert y.shape[2] == 768

    def test_mae_encoder_forward_shape(self):
        encoder = _build_encoder("mae", hidden_size=768, patch_size=16, num_hidden_layers=12).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = _ENCODER_FORWARD_FNS["mae"](encoder, x, patch_size=16)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196  # (224/16)^2
        assert y.shape[2] == 768


@slow
@pytest.mark.skip(reason="Not enough model usage to justify slow tests yet.")
class AutoencoderRAEIntegrationTests:
    def teardown_method(self):
        gc.collect()
        backend_empty_cache(torch_device)

    def test_autoencoder_rae_from_pretrained_dinov2(self):
        model = AutoencoderRAE.from_pretrained("nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08").to(torch_device)
        model.eval()

        image = load_image(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
        )
        image = image.convert("RGB").resize((224, 224))
        x = to_tensor(image).unsqueeze(0).to(torch_device)

        with torch.no_grad():
            latents = model.encode(x).latent
            assert latents.shape == (1, 768, 16, 16)

            recon = model.decode(latents).sample
            assert recon.shape == (1, 3, 256, 256)
            assert torch.isfinite(recon).all().item()

            # fmt: off
            expected_latent_slice = torch.tensor([0.7617, 0.8824, -0.4891])
            expected_recon_slice = torch.tensor([0.1263, 0.1355, 0.1435])
            # fmt: on

            assert torch_all_close(latents[0, :3, 0, 0].float().cpu(), expected_latent_slice, atol=1e-3)
            assert torch_all_close(recon[0, 0, 0, :3].float().cpu(), expected_recon_slice, atol=1e-3)
