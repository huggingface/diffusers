# Copyright 2024 Stability AI and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from math import pi
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class StableAudioPositionalEmbedding(nn.Module):
    """Used for continuous time"""

    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        times = times[..., None]
        freqs = times * self.weights[None] * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((times, fouriered), dim=-1)
        return fouriered


@dataclass
class StableAudioProjectionModelOutput(BaseOutput):
    """
    Args:
    Class for StableAudio projection layer's outputs.
        text_hidden_states (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the hidden-states for the text encoder.
        seconds_start_hidden_states (`torch.Tensor` of shape `(batch_size, 1, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the audio start hidden states.
        seconds_end_hidden_states (`torch.Tensor` of shape `(batch_size, 1, hidden_size)`, *optional*):
            Sequence of hidden-states obtained by linearly projecting the audio end hidden states.
    """

    text_hidden_states: Optional[torch.Tensor] = None
    seconds_start_hidden_states: Optional[torch.Tensor] = None
    seconds_end_hidden_states: Optional[torch.Tensor] = None


class StableAudioNumberConditioner(nn.Module):
    """
    A simple linear projection model to map numbers to a latent space.

    Args:
        number_embedding_dim (`int`):
            Dimensionality of the number embeddings.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
        internal_dim (`int`):
            Dimensionality of the intermediate number hidden states.
    """

    def __init__(
        self,
        number_embedding_dim,
        min_value,
        max_value,
        internal_dim: Optional[int] = 256,
    ):
        super().__init__()
        self.time_positional_embedding = nn.Sequential(
            StableAudioPositionalEmbedding(internal_dim),
            nn.Linear(in_features=internal_dim + 1, out_features=number_embedding_dim),
        )

        self.number_embedding_dim = number_embedding_dim
        self.min_value = min_value
        self.max_value = max_value

    def forward(
        self,
        floats: torch.Tensor,
    ):
        floats = floats.clamp(self.min_value, self.max_value)

        normalized_floats = (floats - self.min_value) / (self.max_value - self.min_value)

        # Cast floats to same type as embedder
        embedder_dtype = next(self.time_positional_embedding.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)

        embedding = self.time_positional_embedding(normalized_floats)
        float_embeds = embedding.view(-1, 1, self.number_embedding_dim)

        return float_embeds


class StableAudioProjectionModel(ModelMixin, ConfigMixin):
    """
    A simple linear projection model to map the conditioning values to a shared latent space.

    Args:
        text_encoder_dim (`int`):
            Dimensionality of the text embeddings from the text encoder (T5).
        conditioning_dim (`int`):
            Dimensionality of the output conditioning tensors.
        min_value (`int`):
            The minimum value of the seconds number conditioning modules.
        max_value (`int`):
            The maximum value of the seconds number conditioning modules
    """

    @register_to_config
    def __init__(self, text_encoder_dim, conditioning_dim, min_value, max_value):
        super().__init__()
        self.text_projection = (
            nn.Identity() if conditioning_dim == text_encoder_dim else nn.Linear(text_encoder_dim, conditioning_dim)
        )
        self.start_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)
        self.end_number_conditioner = StableAudioNumberConditioner(conditioning_dim, min_value, max_value)

    def forward(
        self,
        text_hidden_states: Optional[torch.Tensor] = None,
        start_seconds: Optional[torch.Tensor] = None,
        end_seconds: Optional[torch.Tensor] = None,
    ):
        text_hidden_states = (
            text_hidden_states if text_hidden_states is None else self.text_projection(text_hidden_states)
        )
        seconds_start_hidden_states = (
            start_seconds if start_seconds is None else self.start_number_conditioner(start_seconds)
        )
        seconds_end_hidden_states = end_seconds if end_seconds is None else self.end_number_conditioner(end_seconds)

        return StableAudioProjectionModelOutput(
            text_hidden_states=text_hidden_states,
            seconds_start_hidden_states=seconds_start_hidden_states,
            seconds_end_hidden_states=seconds_end_hidden_states,
        )
