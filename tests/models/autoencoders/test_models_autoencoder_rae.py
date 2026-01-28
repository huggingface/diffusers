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
import os
import unittest

import torch

from diffusers.models.autoencoders.autoencoder_rae import AutoencoderRAE, Dinov2Encoder, MAEEncoder, Siglip2Encoder

from ...testing_utils import backend_empty_cache, enable_full_determinism, slow, torch_device


enable_full_determinism()


def _get_required_local_path(env_name: str) -> str:
    path = os.environ.get(env_name)
    assert path is not None and len(path) > 0, f"Please set `{env_name}` to a local pretrained model directory."
    assert os.path.exists(path), f"Path from `{env_name}` does not exist: {path}"
    return path


@slow
class AutoencoderRAEEncoderIntegrationTests(unittest.TestCase):
    def tearDown(self):
        super().tearDown()
        gc.collect()
        backend_empty_cache(torch_device)

    def test_dinov2_encoder_forward_shape(self):
        dino_path = _get_required_local_path("DINO_PATH")

        encoder = Dinov2Encoder(encoder_name_or_path=dino_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 256
        assert y.shape[2] == encoder.hidden_size

    def test_siglip2_encoder_forward_shape(self):
        siglip2_path = _get_required_local_path("SIGLIP2_PATH")

        encoder = Siglip2Encoder(encoder_name_or_path=siglip2_path).to(torch_device)
        x = torch.rand(1, 3, 224, 224, device=torch_device)
        y = encoder(x)

        assert y.ndim == 3
        assert y.shape[0] == 1
        assert y.shape[1] == 196
        assert y.shape[2] == encoder.hidden_size

    def test_mae_encoder_forward_shape(self):
        mae_path = _get_required_local_path("MAE_PATH")

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
        dino_path = _get_required_local_path("DINO_PATH")

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