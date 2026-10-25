# Copyright 2026 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


from ...loaders import Flux3LoraLoaderMixin
from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Flux3ModularPipeline(ModularPipeline, Flux3LoraLoaderMixin):
    """
    A ModularPipeline for FLUX.3.
    """

    default_blocks_name = "Flux3AutoBlocks"

    @property
    def vae_scale_factor(self):
        vae_scale_factor = 8
        if getattr(self, "vae", None) is not None:
            vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        return vae_scale_factor

    @property
    def num_channels_latents(self):
        num_channels_latents = 128
        if getattr(self, "transformer", None):
            num_channels_latents = self.transformer.config.in_channels
        return num_channels_latents

    @property
    def requires_unconditional_embeds(self):
        if getattr(self, "guider", None) is None:
            return False
        return self.guider._enabled and self.guider.num_conditions > 1


class Flux3DistilledModularPipeline(Flux3ModularPipeline):
    """
    A ModularPipeline for the step-distilled FLUX.3 model (16 steps, no classifier-free guidance).
    """

    default_blocks_name = "Flux3DistilledAutoBlocks"
