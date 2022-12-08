# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from torch import nn

from transformers import CLIPPreTrainedModel, CLIPVisionModel

from ...models.attention import BasicTransformerBlock
from ...utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PaintByExampleImageEncoder(CLIPPreTrainedModel):
    def __init__(self, config, proj_size=768):
        super().__init__(config)
        self.proj_size = proj_size

        self.model = CLIPVisionModel(config)
        self.mapper = PaintByExampleMapper(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, self.proj_size)

        # uncondition for scaling
        self.uncond_vector = nn.Parameter(torch.rand((1, 1, self.proj_size)))

    def forward(self, pixel_values):
        clip_output = self.model(pixel_values=pixel_values)
        latent_states = clip_output.pooler_output
        latent_states = self.mapper(latent_states[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        return latent_states


class PaintByExampleMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        num_layers = (config.num_hidden_layers + 1) // 5
        hid_size = config.hidden_size
        num_heads = 1
        self.blocks = nn.ModuleList(
            [
                BasicTransformerBlock(hid_size, num_heads, hid_size, activation_fn="gelu", attention_bias=True)
                for _ in range(num_layers)
            ]
        )

    def forward(self, hidden_states):
        for block in self.blocks:
            hidden_states = block(hidden_states)

        return hidden_states
