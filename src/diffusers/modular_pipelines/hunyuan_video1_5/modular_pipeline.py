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

    default_blocks_name = "HunyuanVideo15Blocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 16

    @property
    def vae_temporal_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.temporal_compression_ratio
        return 4

    @property
    def num_channels_latents(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.latent_channels
        return 32

    @property
    def target_size(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.target_size
        return 640

    @property
    def default_aspect_ratio(self):
        return (16, 9)

    @property
    def default_height(self):
        from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
        processor = HunyuanVideo15ImageProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        h, w = processor.calculate_default_height_width(
            self.default_aspect_ratio[1], self.default_aspect_ratio[0], self.target_size
        )
        return h

    @property
    def default_width(self):
        from ...pipelines.hunyuan_video1_5.image_processor import HunyuanVideo15ImageProcessor
        processor = HunyuanVideo15ImageProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)
        h, w = processor.calculate_default_height_width(
            self.default_aspect_ratio[1], self.default_aspect_ratio[0], self.target_size
        )
        return w

    @property
    def tokenizer_max_length(self):
        return 1000

    @property
    def tokenizer_2_max_length(self):
        return 256

    @property
    def system_message(self):
        return (
            "You are a helpful assistant. Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video. "
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects. "
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects. "
            "4. background environment, light, style and atmosphere. "
            "5. camera angles, movements, and transitions used in the video."
        )

    @property
    def prompt_template_encode_start_idx(self):
        return 108

    @property
    def vision_num_semantic_tokens(self):
        return 729

    @property
    def vision_states_dim(self):
        if getattr(self, "transformer", None) is not None:
            return self.transformer.config.image_embed_dim
        return 1152


