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

from diffusers import LongCatAudioDiTVae


def test_longcat_audio_vae_default_strides_match_downsampling_ratio():
    vae = LongCatAudioDiTVae(channels=1, latent_dim=2, encoder_latent_dim=4)

    assert vae.config.strides == [2, 4, 4, 8, 8]
    assert vae.config.downsampling_ratio == 2048


def test_longcat_audio_vae_raises_when_downsampling_ratio_mismatches_strides():
    with pytest.raises(ValueError, match="downsampling_ratio"):
        LongCatAudioDiTVae(
            channels=1,
            latent_dim=2,
            encoder_latent_dim=4,
            strides=[2, 2, 2, 2, 2],
            downsampling_ratio=2048,
        )
