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

from typing import Tuple

import torch
from torch import nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models import ModelMixin


class ChannelsProj(nn.Module):
    def __init__(
        self,
        *,
        vectors: int,
        channels: int,
        d_latent: int,
    ):
        super().__init__()
        self.proj = nn.Linear(d_latent, vectors * channels)
        self.norm = nn.LayerNorm(channels)
        self.d_latent = d_latent
        self.vectors = vectors
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bvd = x
        w_vcd = self.proj.weight.view(self.vectors, self.channels, self.d_latent)
        b_vc = self.proj.bias.view(1, self.vectors, self.channels)
        h = torch.einsum("bvd,vcd->bvc", x_bvd, w_vcd)
        h = self.norm(h)

        h = h + b_vc
        return h


class ShapEParamsProjModel(ModelMixin, ConfigMixin):
    """
    project the latent representation of a 3D asset to obtain weights of a multi-layer perceptron (MLP).

    For more details, see the original paper:
    """

    @register_to_config
    def __init__(
        self,
        *,
        param_names: Tuple[str] = (
            "nerstf.mlp.0.weight",
            "nerstf.mlp.1.weight",
            "nerstf.mlp.2.weight",
            "nerstf.mlp.3.weight",
        ),
        param_shapes: Tuple[Tuple[int]] = (
            (256, 93),
            (256, 256),
            (256, 256),
            (256, 256),
        ),
        d_latent: int = 1024,
    ):
        super().__init__()

        # check inputs
        if len(param_names) != len(param_shapes):
            raise ValueError("Must provide same number of `param_names` as `param_shapes`")
        self.projections = nn.ModuleDict({})
        for k, (vectors, channels) in zip(param_names, param_shapes):
            self.projections[_sanitize_name(k)] = ChannelsProj(
                vectors=vectors,
                channels=channels,
                d_latent=d_latent,
            )

    def forward(self, x: torch.Tensor):
        out = {}
        start = 0
        for k, shape in zip(self.config.param_names, self.config.param_shapes):
            vectors, _ = shape
            end = start + vectors
            x_bvd = x[:, start:end]
            out[k] = self.projections[_sanitize_name(k)](x_bvd).reshape(len(x), *shape)
            start = end
        return out


def _sanitize_name(x: str) -> str:
    return x.replace(".", "__")
