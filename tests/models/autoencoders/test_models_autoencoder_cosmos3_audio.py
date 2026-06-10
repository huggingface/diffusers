# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import importlib.util
from pathlib import Path

import pytest
import torch

from diffusers.models.autoencoders.autoencoder_cosmos3_audio import (
    Cosmos3AudioDiagonalGaussianDistribution,
    Cosmos3AVAEAudioTokenizer,
)


def _get_tiny_cosmos3_audio_tokenizer() -> Cosmos3AVAEAudioTokenizer:
    return Cosmos3AVAEAudioTokenizer(
        sampling_rate=16,
        hop_size=4,
        input_channels=1,
        stereo=True,
        normalize_volume=True,
        enc_dim=4,
        enc_num_blocks=1,
        enc_n_fft=8,
        enc_hop_length=2,
        enc_latent_dim=8,
        enc_c_mults=(1,),
        enc_strides=(2,),
        vocoder_input_dim=4,
        dec_dim=4,
        dec_c_mults=(1, 2),
        dec_strides=(2, 2),
        dec_out_channels=2,
    )


def test_cosmos3_audio_tokenizer_encode_decode_forward_shapes():
    torch.manual_seed(0)
    model = _get_tiny_cosmos3_audio_tokenizer().eval()
    audio = torch.randn(2, 2, 15)

    encoded = model.encode(audio)
    assert isinstance(encoded.latent_dist, Cosmos3AudioDiagonalGaussianDistribution)
    assert encoded.latent_dist.mean.shape == (2, 4, 4)
    assert encoded.latent_dist.scale.shape == (2, 4, 4)

    latents = encoded.latent_dist.mode()
    decoded = model.decode(latents)
    assert decoded.shape == (2, 2, 16)
    assert decoded.min() >= -1.0
    assert decoded.max() <= 1.0

    forward_output = model(audio)
    assert forward_output.sample.shape == (2, 2, 16)

    tuple_output = model(audio, return_dict=False)
    assert tuple_output[0].shape == (2, 2, 16)


def test_cosmos3_audio_tokenizer_encode_tuple_and_seeded_sample():
    torch.manual_seed(0)
    model = _get_tiny_cosmos3_audio_tokenizer().eval()
    audio = torch.randn(1, 2, 16)

    posterior = model.encode(audio, return_dict=False)[0]
    sample_a = posterior.sample(generator=torch.Generator("cpu").manual_seed(13))
    sample_b = posterior.sample(generator=torch.Generator("cpu").manual_seed(13))

    assert torch.allclose(sample_a, sample_b)
    assert sample_a.shape == (1, 4, 4)
    assert posterior.kl().ndim == 0


def test_cosmos3_audio_tokenizer_decoder_only_state_disables_encode():
    model = _get_tiny_cosmos3_audio_tokenizer()
    decoder_only_state_dict = {key: value for key, value in model.state_dict().items() if key.startswith("decoder.")}

    decoder_only_model = _get_tiny_cosmos3_audio_tokenizer()
    decoder_only_model._fix_state_dict_keys_on_load(decoder_only_state_dict)
    decoder_only_model.load_state_dict(decoder_only_state_dict, strict=True)

    assert decoder_only_model.encoder is None
    with pytest.raises(ValueError, match="decoder-only weights"):
        decoder_only_model.encode(torch.randn(1, 2, 16))


def _load_converter_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "convert_cosmos3_to_diffusers.py"
    spec = importlib.util.spec_from_file_location("convert_cosmos3_to_diffusers", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cosmos3_audio_converter_keeps_encoder_and_remaps_decoder():
    converter = _load_converter_module()
    state_dict = {
        "generator.encoder.layers.0.weight": torch.ones(4, 20, 1),
        "generator.encoder.layers.1.act.alpha": torch.zeros(16),
        "generator.encoder.layers.1.act.beta": torch.zeros(16),
        "generator.decoder.layers.0.weight": torch.ones(8, 4, 7),
        "generator.decoder.layers.1.layers.0.alpha": torch.zeros(8),
        "generator.decoder.layers.1.layers.1.weight": torch.ones(8, 4, 4),
        "generator.decoder.layers.1.layers.2.layers.0.alpha": torch.zeros(4),
        "generator.decoder.layers.1.layers.2.layers.1.weight": torch.ones(4, 4, 7),
        "generator.decoder.layers.2.alpha": torch.zeros(4),
        "generator.decoder.layers.3.weight": torch.ones(2, 4, 7),
    }

    remapped = converter._remap_avae_state_dict(state_dict)

    assert not any(key.startswith("decoder.layers.") for key in remapped)
    assert "encoder.layers.0.weight" not in remapped
    assert "encoder.layers.0.weight_g" in remapped
    assert "encoder.layers.0.weight_v" in remapped
    assert remapped["encoder.layers.1.act.alpha"].shape == (16,)
    assert remapped["decoder.conv1.weight_g"].shape == (8, 1, 1)
    assert remapped["decoder.block.0.snake1.alpha"].shape == (1, 8, 1)
    assert remapped["decoder.block.0.res_unit1.snake1.alpha"].shape == (1, 4, 1)
    assert remapped["decoder.snake1.alpha"].shape == (1, 4, 1)


def test_cosmos3_audio_converter_allows_decoder_only_state_dict():
    converter = _load_converter_module()
    state_dict = {
        "decoder.conv1.weight": torch.ones(8, 4, 7),
        "decoder.snake1.alpha": torch.zeros(4),
    }

    remapped = converter._remap_avae_state_dict(state_dict)

    assert not any(key.startswith("encoder.") for key in remapped)
    assert "decoder.conv1.weight_g" in remapped
    assert remapped["decoder.snake1.alpha"].shape == (1, 4, 1)
