# Copyright 2026 Pruna AI and The HuggingFace Team. All rights reserved.
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
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin


class PH3ImageTextConnector(ModelMixin, ConfigMixin):
    r"""
    Maps hidden states of P-H3-Image's Qwen3-VL text encoder to the text conditioning its transformer reads.

    The hidden states listed in `hidden_state_indices` are concatenated, standardized, projected linearly and
    destandardized. The first token is an attention sink with the same conditioning for every prompt, so it is replaced
    by `sink_embedding`.

    Args:
        hidden_state_indices (`tuple[int, ...]`, defaults to `(13, 19, 25, 31, 36)`):
            Which entries of the text encoder's `hidden_states` are read, in the numbering of `output_hidden_states`
            (`0` is the embedding output, `i` the output of decoder layer `i`, and the last one the final normed hidden
            state).
        text_hidden_size (`int`, defaults to `2560`):
            The hidden size of the text encoder.
        out_channels (`int`, defaults to `5120`):
            The number of channels of the conditioning MiniMax-H3's transformer reads.
    """

    # Float32, as in the released checkpoint.
    _keep_in_fp32_modules = ["input_mean", "input_std", "proj", "output_mean", "output_std", "sink_embedding"]

    @register_to_config
    def __init__(
        self,
        hidden_state_indices: tuple[int, ...] = (13, 19, 25, 31, 36),
        text_hidden_size: int = 2560,
        out_channels: int = 5120,
    ) -> None:
        super().__init__()
        in_channels = len(hidden_state_indices) * text_hidden_size

        self.input_mean = nn.Parameter(torch.zeros(in_channels))
        self.input_std = nn.Parameter(torch.ones(in_channels))
        self.proj = nn.Linear(in_channels, out_channels, bias=False)
        self.output_mean = nn.Parameter(torch.zeros(out_channels))
        self.output_std = nn.Parameter(torch.ones(out_channels))
        self.sink_embedding = nn.Parameter(torch.zeros(out_channels))

    def forward(self, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`tuple[torch.Tensor, ...]`):
                The hidden states of the text encoder, as `output_hidden_states=True` returns them: one tensor of shape
                `(batch_size, num_tokens, text_hidden_size)` per entry.

        Returns:
            `torch.Tensor`: the conditioning, of shape `(batch_size, num_tokens, out_channels)`, in the dtype of the
            connector's parameters.
        """
        dtype = self.dtype
        hidden_states = torch.cat([hidden_states[index] for index in self.config.hidden_state_indices], dim=-1)
        hidden_states = (hidden_states.to(dtype) - self.input_mean) / self.input_std
        hidden_states = self.proj(hidden_states) * self.output_std + self.output_mean
        sink = self.sink_embedding.expand(hidden_states.shape[0], 1, -1)
        return torch.cat([sink, hidden_states[:, 1:]], dim=1)
