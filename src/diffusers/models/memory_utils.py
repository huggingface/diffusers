# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from typing import Optional

import torch

from ..utils import logging
from .activations import GEGLU, GELU, ApproximateGELU, LinearActivation, SwiGLU
from .attention import FeedForward


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class _MemoryOptimizedFeedForward(torch.nn.Module):
    r"""
    See [`~models.attention.FeedForward`] parameter documentation. This class is a copy of the FeedForward class. The
    only difference is that this module is optimized for memory.

    This method achieves memory savings by applying the ideas of tensor-parallelism sequentially. Input projection
    layers are split column-wise and output projection layers are split row-wise. This allows for the computation of
    the feedforward pass to occur without ever materializing the full intermediate tensor. Typically, the intermediate
    tensor takes 4x-8x more memory than the input tensor. This method reduces that with a small performance tradeoff.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        num_splits: int = 4,
    ) -> None:
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)

        dim_out = dim_out if dim_out is not None else dim

        dim_split = inner_dim // num_splits
        if inner_dim % dim_split != 0:
            raise ValueError(f"inner_dim must be divisible by {mult=}, or {num_splits=} if provided.")

        self._dim = dim
        self._dim_out = dim_out
        self._mult = mult
        self._dropout = dropout
        self._activation_fn = activation_fn
        self._final_dropout = final_dropout
        self._inner_dim = inner_dim
        self._bias = bias
        self._num_splits = num_splits

        def get_activation_fn(dim_: int, inner_dim_: int):
            if activation_fn == "gelu":
                act_fn = GELU(dim_, inner_dim_, bias=bias)
            if activation_fn == "gelu-approximate":
                act_fn = GELU(dim_, inner_dim_, approximate="tanh", bias=bias)
            elif activation_fn == "geglu":
                act_fn = GEGLU(dim_, inner_dim_, bias=bias)
            elif activation_fn == "geglu-approximate":
                act_fn = ApproximateGELU(dim_, inner_dim_, bias=bias)
            elif activation_fn == "swiglu":
                act_fn = SwiGLU(dim_, inner_dim_, bias=bias)
            elif activation_fn == "linear-silu":
                act_fn = LinearActivation(dim_, inner_dim_, bias=bias, activation="silu")
            return act_fn

        # Split column-wise
        self.proj_in = torch.nn.ModuleList([get_activation_fn(dim, dim_split) for _ in range(inner_dim // dim_split)])

        self.dropout = torch.nn.Dropout(dropout)

        # Split row-wise
        self.proj_out = torch.nn.ModuleList(
            [torch.nn.Linear(dim_split, dim_out, bias=False) for _ in range(inner_dim // dim_split)]
        )

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out))

        self.final_dropout = None
        if final_dropout:
            self.final_dropout = torch.nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Output tensor for "all_reduce" operation
        output = hidden_states.new_zeros(hidden_states.shape)

        # Apply feedforward pass sequentially since this is intended for memory optimization on a single GPU
        for proj_in, proj_out in zip(self.proj_in, self.proj_out):
            out = proj_in(hidden_states)
            out = self.dropout(out)
            out = proj_out(out)
            # Perform "all_reduce"
            output += out

        if self.bias is not None:
            output += self.bias
        if self.final_dropout is not None:
            output = self.final_dropout(output)

        return output


def apply_memory_optimized_feedforward(module: torch.nn.Module, num_splits: Optional[int] = None) -> torch.nn.Module:
    module_dict = dict(module.named_modules())

    for name, submodule in module_dict.items():
        if not isinstance(submodule, FeedForward):
            continue

        logger.debug(f"Applying memory optimized feedforward to layer '{name}'")
        state_dict = submodule.state_dict()
        num_splits = submodule._mult if num_splits is None else num_splits

        # remap net.0.proj.weight
        net_0_proj = state_dict.pop("net.0.proj.weight")
        net_0_proj = net_0_proj.chunk(num_splits, dim=0)
        for i in range(num_splits):
            state_dict[f"proj_in.{i}.proj.weight"] = net_0_proj[i]

        # remap net.0.proj.bias
        if "net.0.proj.bias" in state_dict:
            net_0_proj_bias = state_dict.pop("net.0.proj.bias")
            net_0_proj_bias = net_0_proj_bias.chunk(num_splits, dim=0)
            for i in range(num_splits):
                state_dict[f"proj_in.{i}.proj.bias"] = net_0_proj_bias[i]

        # remap net.2.weight
        net_2_weight = state_dict.pop("net.2.weight")
        net_2_weight = net_2_weight.chunk(num_splits, dim=1)
        for i in range(num_splits):
            state_dict[f"proj_out.{i}.weight"] = net_2_weight[i]

        # remap net.2.bias
        if "net.2.bias" in state_dict:
            net_2_bias = state_dict.pop("net.2.bias")
            state_dict["bias"] = net_2_bias

        with torch.device("meta"):
            new_ff = _MemoryOptimizedFeedForward(
                dim=submodule._dim,
                dim_out=submodule._dim_out,
                mult=submodule._mult,
                dropout=submodule._dropout,
                activation_fn=submodule._activation_fn,
                final_dropout=submodule._final_dropout,
                inner_dim=submodule._inner_dim,
                bias=submodule._bias,
                num_splits=num_splits,
            )

        new_ff.load_state_dict(state_dict, strict=True, assign=True)

        parent_module_name, _, submodule_name = name.rpartition(".")
        parent_module = module_dict[parent_module_name]
        setattr(parent_module, submodule_name, new_ff)

    return module
