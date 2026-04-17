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

from ...loaders import HunyuanVideoLoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)


class HunyuanVideo15ModularPipeline(
    ModularPipeline,
    HunyuanVideoLoraLoaderMixin,
):
    """
    A ModularPipeline for HunyuanVideo 1.5.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "HunyuanVideo15AutoBlocks"

    @property
    def vae_scale_factor_spatial(self):
        return self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16

    @property
    def vae_scale_factor_temporal(self):
        return self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4

    @property
    def num_channels_latents(self):
        return self.vae.config.latent_channels if getattr(self, "vae", None) else 32

    @property
    def target_size(self):
        return self.transformer.config.target_size if getattr(self, "transformer", None) else 640

    @property
    def default_aspect_ratio(self):
        return (16, 9)

    @property
    def vision_num_semantic_tokens(self):
        return 729

    @property
    def vision_states_dim(self):
        return self.transformer.config.image_embed_dim if getattr(self, "transformer", None) else 1152

    @property
    def tokenizer_max_length(self):
        return 1000

    @property
    def tokenizer_2_max_length(self):
        return 256

    # fmt: off
    @property
    def system_message(self):
        return "You are a helpful assistant. Describe the video by detailing the following aspects: \
        1. The main content and theme of the video. \
        2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. \
        3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. \
        4. background environment, light, style and atmosphere. \
        5. camera angles, movements, and transitions used in the video."
    # fmt: on

    @property
    def prompt_template_encode_start_idx(self):
        return 108

    @property
    def requires_unconditional_embeds(self):
        if hasattr(self, "guider") and self.guider is not None:
            return self.guider._enabled and self.guider.num_conditions > 1
        return False
