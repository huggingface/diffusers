# Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.
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

"""LM head that turns the head-less Qwen3-VL text encoder of Ideogram 4 into a generative model."""

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ..modeling_utils import ModelMixin


class Ideogram4PromptEnhancerHead(ModelMixin, ConfigMixin):
    """LM head that makes the head-less Qwen3-VL `text_encoder` generative for prompt upsampling.

    An optional pipeline component (`prompt_enhancer_head`): its weights load via a normal `from_pretrained` (its own
    small repo, or bundled in the model repo) rather than an in-pipeline download. At upsample time the pipeline
    combines it with the shared `text_encoder` body to form the generative model.
    """

    @register_to_config
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 151936):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states (`torch.Tensor`):
                Final hidden states of the text encoder, of shape `(batch_size, sequence_length, hidden_size)`.

        Returns:
            `torch.Tensor`:
                Next-token logits of shape `(batch_size, sequence_length, vocab_size)`.
        """
        return self.lm_head(hidden_states)
