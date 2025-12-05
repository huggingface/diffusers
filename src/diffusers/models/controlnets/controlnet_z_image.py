# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
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

from typing import List, Optional

import torch
import torch.nn as nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin
from ..controlnets.controlnet import zero_module
from ..modeling_utils import ModelMixin
from ..transformers.transformer_z_image import (
    ZImageTransformerBlock,
)


class ZImageControlTransformerBlock(ZImageTransformerBlock):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        block_id=0,
    ):
        super().__init__(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = zero_module(nn.Linear(self.dim, self.dim))
        self.after_proj = zero_module(nn.Linear(self.dim, self.dim))

    def forward(
        self,
        c: torch.Tensor,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
    ):
        if self.block_id == 0:
            c = self.before_proj(c) + x
            all_c = []
        else:
            all_c = list(torch.unbind(c))
            c = all_c.pop(-1)

        c = super().forward(c, attn_mask, freqs_cis, adaln_input)
        c_skip = self.after_proj(c)
        all_c += [c_skip, c]
        c = torch.stack(all_c)
        return c


class ZImageControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        dim=3840,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        control_layers_places: List[int] = None,
        control_in_dim=None,
    ):
        super().__init__()
        self.control_layers_places = control_layers_places
        self.control_in_dim = control_in_dim

        assert 0 in self.control_layers_places

        # control blocks
        self.control_layers = nn.ModuleList(
            [
                ZImageControlTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, qk_norm, block_id=i)
                for i in self.control_layers_places
            ]
        )

        # control patch embeddings
        all_x_embedder = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * self.control_in_dim, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

        self.control_all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.control_noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

    def forward(self):
        pass
