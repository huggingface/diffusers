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

from ...loaders import LTX2LoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class LTX2ModularPipeline(
    ModularPipeline,
    LTX2LoraLoaderMixin,
):
    """
    A ModularPipeline for LTX-2 (joint video + audio generation).

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LTX2AutoBlocks"

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
    def audio_sampling_rate(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sample_rate
        return 16000

    @property
    def audio_hop_length(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.mel_hop_length
        return 160
