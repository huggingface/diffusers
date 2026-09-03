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

import torch

from ...loaders import LTX2LoraLoaderMixin as EchoLoraLoaderMixin
from ..modular_pipeline import ModularPipeline
from .echo_constants import (
    ECHO_AUDIO_LATENTS_MEAN,
    ECHO_AUDIO_LATENTS_STD,
    ECHO_LATENTS_MEAN,
    ECHO_LATENTS_STD,
)


class EchoModularPipeline(ModularPipeline, EchoLoraLoaderMixin):
    """A ModularPipeline for the Echo long-video DMD checkpoint."""

    default_blocks_name = "EchoBlocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 32

    @property
    def vae_temporal_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.temporal_compression_ratio
        return 8

    @property
    def vae_scaling_factor(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.scaling_factor
        return 1.0

    @property
    def latents_mean(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.latents_mean
        return torch.tensor(ECHO_LATENTS_MEAN)

    @property
    def latents_std(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.latents_std
        return torch.tensor(ECHO_LATENTS_STD)

    @property
    def transformer_spatial_patch_size(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.patch_size
        return 1

    @property
    def transformer_temporal_patch_size(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.patch_size_t
        return 1

    @property
    def audio_vae_mel_compression_ratio(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.mel_compression_ratio
        return 4

    @property
    def audio_vae_temporal_compression_ratio(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.temporal_compression_ratio
        return 4

    @property
    def audio_latents_mean(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.latents_mean
        return torch.tensor(ECHO_AUDIO_LATENTS_MEAN)

    @property
    def audio_latents_std(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.latents_std
        return torch.tensor(ECHO_AUDIO_LATENTS_STD)

    @property
    def audio_sampling_rate(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sample_rate
        return 16000

    @property
    def audio_hop_length(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.mel_hop_length
        return 160
