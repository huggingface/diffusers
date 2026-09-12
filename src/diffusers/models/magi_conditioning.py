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

from dataclasses import dataclass

import torch
from torch import nn

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from .modeling_utils import ModelMixin


@dataclass
class MagiTextConditioningOutput(BaseOutput):
    sample: torch.Tensor
    attention_mask: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    negative_prompt_attention_mask: torch.Tensor


class MagiTextConditioningModel(ModelMixin, ConfigMixin):
    """Store MAGI's learned null caption and the official HQ/duration feature vectors."""

    _no_split_modules = ["MagiTextConditioningModel"]
    _keep_in_fp32_modules = ["null_embedding", "special_embedding"]

    @register_to_config
    def __init__(self, caption_channels=4096, caption_max_length=800, null_token_length=50):
        super().__init__()
        if not 0 < null_token_length <= caption_max_length:
            raise ValueError("null_token_length must be between one and caption_max_length.")
        self.null_embedding = nn.Embedding(caption_max_length, caption_channels)
        self.special_embedding = nn.Embedding(9, caption_channels)
        nn.init.zeros_(self.null_embedding.weight)
        nn.init.zeros_(self.special_embedding.weight)

    def forward(self, hidden_states, attention_mask, num_chunks=1, return_dict=True):
        """
        Prepare chunk-dependent conditional and stationary null text features.

        Args:
            hidden_states (`torch.Tensor`): T5 features shaped `(batch, length, caption_channels)`.
            attention_mask (`torch.Tensor`): Boolean keep-mask shaped `(batch, length)`.
            num_chunks (`int`, defaults to `1`): Number of video chunks to condition.
            return_dict (`bool`, defaults to `True`): Return structured conditioning outputs.

        Returns:
            `MagiTextConditioningOutput` or `tuple`: Conditional features and masks, followed by null features and
            masks.
        """
        batch, length, channels = hidden_states.shape
        if (length, channels) != (self.config.caption_max_length, self.config.caption_channels):
            raise ValueError("T5 features must match the conditioning model's caption length and channels.")
        if attention_mask.shape != (batch, length) or num_chunks < 1:
            raise ValueError("Expected a matching text mask and at least one chunk.")
        indices = torch.arange(length, device=hidden_states.device)
        null = self.null_embedding(indices).to(hidden_states.dtype)[None].expand(batch, -1, -1)
        hq = self.special_embedding(torch.zeros(1, device=hidden_states.device, dtype=torch.long))
        duration_indices = torch.arange(num_chunks, 0, -1, device=hidden_states.device).clamp(max=8)
        duration = self.special_embedding(duration_indices).to(hidden_states.dtype)
        conditional = torch.cat(
            [
                duration[None, :, None].expand(batch, -1, -1, -1),
                hq.to(hidden_states.dtype)[None, None].expand(batch, num_chunks, -1, -1),
                hidden_states[:, None].expand(-1, num_chunks, -1, -1),
            ],
            dim=2,
        )[:, :, :length]
        mask = torch.cat(
            [
                torch.ones(batch, num_chunks, 2, device=hidden_states.device, dtype=torch.bool),
                attention_mask.bool()[:, None].expand(-1, num_chunks, -1),
            ],
            dim=2,
        )[:, :, :length]
        null_mask = (indices < self.config.null_token_length)[None].expand(batch, -1)
        if not return_dict:
            return conditional, mask, null, null_mask
        return MagiTextConditioningOutput(conditional, mask, null, null_mask)
