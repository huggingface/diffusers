# Copyright 2025 The HuggingFace Team. All rights reserved.
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


from ...loaders import LTXVideoLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class LTXModularPipeline(
    ModularPipeline,
    LTXVideoLoraLoaderMixin,
):
    """
    A ModularPipeline for LTX Video.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "LTXBlocks"

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
    def requires_unconditional_embeds(self):
        if hasattr(self, "guider") and self.guider is not None:
            return self.guider._enabled and self.guider.num_conditions > 1
        return False
