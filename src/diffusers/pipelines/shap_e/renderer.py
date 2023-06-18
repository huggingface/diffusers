# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import math

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin

def posenc_nerf(x: torch.Tensor, min_deg: int = 0, max_deg: int = 15) -> torch.Tensor:
    """
    Concatenate x and its positional encodings, following NeRF.

    Reference: https://arxiv.org/pdf/2210.04628.pdf
    """
    if min_deg == max_deg:
        return x
    scales = 2.0 ** torch.arange(min_deg, max_deg, dtype=x.dtype)
    *shape, dim = x.shape
    xb = (x.reshape(-1, 1, dim) * scales.view(1, -1, 1)).reshape(*shape, -1)
    assert xb.shape[-1] == dim * (max_deg - min_deg)
    emb = torch.cat([xb, xb + math.pi / 2.0], axis=-1).sin()
    return torch.cat([x, emb], dim=-1)

def encode_position(position):

    return posenc_nerf(position, min_deg=0, max_deg=15)

def encode_direction(position, direction=None):
    if direction is None:
        return torch.zeros_like(posenc_nerf(position, min_deg=0, max_deg=8))
    else:
        return posenc_nerf(direction, min_deg=0, max_deg=8)

class MLPNeRSTFModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        d_hidden: int = 256,
        n_output: int = 12,
        n_hidden_layers: int = 6,
        act_fn: str = "swish",
        insert_direction_at: int = 4,

    ):  
        super().__init__()
        # Instantiate the MLP
        
        # Find out the dimension of encoded position and direction 
        dummy = torch.eye(1, 3)
        d_posenc_pos = encode_position(position=dummy).shape[-1]
        d_posenc_dir = encode_direction(position=dummy).shape[-1]

        mlp_widths = [d_hidden] * n_hidden_layers
        input_widths = [d_posenc_pos] + mlp_widths
        output_widths = mlp_widths + [n_output]

        if insert_direction_at is not None:
            input_widths[insert_direction_at] += d_posenc_dir
    
        self.mlp = nn.ModuleList(
                [
                    nn.Linear(d_in, d_out) for d_in, d_out in zip(input_widths, output_widths)
                ]
            )

        if act_fn == "swish":
            self.activation = lambda x: torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation function {act_fn}")
        
    
    def forward(self, *, positions, directions):

        h = encode_position(position)
        h_preact = h
        h_directionless = None
        for i, layer in enumerate(self.mlp):
            if i == self.config.insert_direction_at: # 4 in the config 
                h_directionless = h_preact
                h_direction = encode_direction(position, direction=direction)
                h = torch.cat([h, h_direction], dim=-1)

            h = layer(h)
            h_preact = h
            if i < len(self.mlp) - 1:
                h = self.activation(h)
        h_final = h
        if h_directionless is None:
            h_directionless = h_preact
        return h_final, h_directionless