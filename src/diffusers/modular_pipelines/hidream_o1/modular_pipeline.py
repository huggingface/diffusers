# Copyright 2026 chinoll and The HuggingFace Team. All rights reserved.
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

from ...configuration_utils import ConfigMixin, register_to_config
from ..modular_pipeline import ModularPipeline
from .utils import PATCH_SIZE


class HiDreamO1Patchifier(ConfigMixin):
    """
    Pack and unpack raw RGB image patches for HiDream-O1.
    """

    config_name = "config.json"

    @register_to_config
    def __init__(self, patch_size: int = PATCH_SIZE):
        super().__init__()

    def pack_image(self, image: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = image.shape
        patch_size = self.config.patch_size
        image = image.reshape(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        image = image.permute(0, 2, 4, 1, 3, 5)
        return image.reshape(batch_size, -1, channels * patch_size * patch_size)

    def unpack_image(self, patches: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, patch_dim = patches.shape
        patch_size = self.config.patch_size
        channels = patch_dim // (patch_size * patch_size)
        height_patches = height // patch_size
        width_patches = width // patch_size
        patches = patches.reshape(batch_size, height_patches, width_patches, channels, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(batch_size, channels, height, width)


class HiDreamO1ModularPipeline(ModularPipeline):
    """
    Modular pipeline for HiDream-O1 text-to-image generation.

    HiDream-O1 predicts raw RGB image patches directly and therefore does not use a VAE.
    """

    default_blocks_name = "HiDreamO1AutoBlocks"

    @property
    def default_height(self):
        return self.default_sample_size

    @property
    def default_width(self):
        return self.default_sample_size

    @property
    def default_sample_size(self):
        return 2048
