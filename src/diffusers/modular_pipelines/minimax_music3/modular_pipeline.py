# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

from ...utils import logging
from ..modular_pipeline import ModularPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxMusic3ModularPipeline(ModularPipeline):
    """
    A ModularPipeline for lyrics- and caption-conditioned music generation with MiniMax Music 3.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxMusic3Blocks"

    @property
    def sampling_rate(self):
        sampling_rate = 44100
        if hasattr(self, "vocoder") and self.vocoder is not None:
            sampling_rate = int(self.vocoder.config.sampling_rate)
        return sampling_rate

    @property
    def frame_rate(self):
        # Frames per second of the autoregressive stage (25 Hz for the released checkpoint).
        frame_rate = 25.0
        if hasattr(self, "condition_encoder") and self.condition_encoder is not None:
            config = self.condition_encoder.config
            frame_rate = config.input_sampling_rate / config.input_hop_length
        return frame_rate

    @property
    def latent_hop_length(self):
        # Waveform samples per Flow-VAE latent frame.
        latent_hop_length = 512
        if hasattr(self, "condition_encoder") and self.condition_encoder is not None:
            latent_hop_length = int(self.condition_encoder.config.output_hop_length)
        return latent_hop_length

    @property
    def num_channels_latents(self):
        num_channels_latents = 128
        if hasattr(self, "transformer") and self.transformer is not None:
            num_channels_latents = self.transformer.config.in_channels
        return num_channels_latents
