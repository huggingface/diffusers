# Copyright 2025 Stability AI and The HuggingFace Team. All rights reserved.
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

"""
Auxiliary modules for the Stable Audio 3 pipeline.

``StableAudio3DurationEmbedder`` mirrors the ``NumberConditioner`` used in the reference implementation of SA3
(``stable_audio_tools``). It encodes a duration value (in seconds) into a fixed-dimensional vector that is passed to
the DiT as global conditioning (AdaLN).

Architecture (matches reference ``conditioners.py`` ``NumberConditioner`` with ``fourier_features_type="expo"``):

    1. Normalize: ``t_norm = clamp(t, min_val, max_val); t_norm = (t_norm - min_val) / (max_val - min_val)``
    2. ``ExpoFourierFeatures``: fixed exponentially-spaced Fourier basis (ref:
       ``stable_audio_tools.models.blocks.ExpoFourierFeatures``)
    3. Linear projection: ``fourier_dim → output_dim``
"""

import math

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...utils import logging


logger = logging.get_logger(__name__)


class StableAudio3DurationEmbedder(ModelMixin, ConfigMixin):
    """
    Embeds a duration value (in seconds) into a global conditioning vector for the Stable Audio 3 DiT (used as the
    ``global_hidden_states`` AdaLN input).

    Replicates ``NumberConditioner(fourier_features_type="expo")`` from the SA3 reference implementation.

    Args:
        output_dim (`int`, defaults to 768):
            Dimension of the output embedding. Must match the DiT's ``global_cond_dim``.
        fourier_dim (`int`, defaults to 256):
            Internal Fourier feature dimension (must be even).
        min_val (`float`, defaults to 0.0):
            Minimum duration value for normalization clamping.
        max_val (`float`, defaults to 384.0):
            Maximum duration value for normalization clamping. Values above this are clamped. 384 seconds is the
            production SA3 Medium upper bound for the ``seconds_total`` conditioner.
        min_freq (`float`, defaults to 0.5):
            Minimum frequency for the exponential Fourier basis.
        max_freq (`float`, defaults to 10000.0):
            Maximum frequency for the exponential Fourier basis.
    """

    @register_to_config
    def __init__(
        self,
        output_dim: int = 768,
        fourier_dim: int = 256,
        min_val: float = 0.0,
        max_val: float = 384.0,
        min_freq: float = 0.5,
        max_freq: float = 10000.0,
    ) -> None:
        super().__init__()

        # ExpoFourierFeatures — fixed (no learnable parameters)
        # Frequencies are exponentially spaced from min_freq to max_freq.
        half = fourier_dim // 2
        ramp = torch.linspace(0.0, 1.0, half)
        log_min = math.log(min_freq)
        log_max = math.log(max_freq)
        freqs = torch.exp(ramp * (log_max - log_min) + log_min)
        self.register_buffer("freqs", freqs)

        # Linear projection: fourier_dim → output_dim
        self.linear = nn.Linear(fourier_dim, output_dim)

    def forward(self, seconds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seconds (`torch.Tensor` of shape `(batch,)`):
                Duration values in seconds.

        Returns:
            `torch.Tensor` of shape `(batch, output_dim)` — duration embeddings.
        """
        # 1. Normalize to [0, 1]
        seconds = seconds.float().clamp(self.config.min_val, self.config.max_val)
        t_norm = (seconds - self.config.min_val) / (self.config.max_val - self.config.min_val)

        # 2. Exponential Fourier features — run in fp32 for stability
        t_norm = t_norm.reshape(-1, 1)  # (B, 1)
        args = t_norm * self.freqs.unsqueeze(0) * 2.0 * math.pi  # (B, half)
        fourier = torch.cat([args.cos(), args.sin()], dim=-1)  # (B, fourier_dim)

        # 3. Linear projection
        return self.linear(fourier.to(self.dtype))  # (B, output_dim)
