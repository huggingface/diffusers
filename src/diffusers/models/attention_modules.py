# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import inspect
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import logging
from ..utils.torch_utils import maybe_allow_in_graph
from .attention_processor import (
    AttentionModuleMixin,
    AttnProcessorSDPA,
    FluxAttnProcessorSDPA,
    FusedFluxAttnProcessorSDPA,
    JointAttnProcessorSDPA,
    FusedJointAttnProcessorSDPA,
    SanaLinearAttnProcessorSDPA,
)
from .normalization import RMSNorm, get_normalization


logger = logging.get_logger(__name__)


@maybe_allow_in_graph
class SanaAttention(nn.Module, AttentionModuleMixin):
    """
    Attention implementation specialized for Sana models.

    This module implements lightweight multi-scale linear attention as used in Sana.

    Args:
        in_channels (`int`): Number of input channels.
        out_channels (`int`): Number of output channels.
        num_attention_heads (`int`, *optional*): Number of attention heads.
        attention_head_dim (`int`, defaults to 8): Dimension of each attention head.
        mult (`float`, defaults to 1.0): Multiplier for inner dimension.
        norm_type (`str`, defaults to "batch_norm"): Type of normalization.
        kernel_sizes (`Tuple[int, ...]`, defaults to (5,)): Kernel sizes for multi-scale attention.
    """

    # Set Sana-specific processor classes
    default_processor_class = SanaLinearAttnProcessorSDPA
    fused_processor_class = None  # Sana doesn't have a fused processor yet

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: Optional[int] = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: Tuple[int, ...] = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):
        super().__init__()

        # Core parameters
        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        # Calculate dimensions
        num_attention_heads = (
            int(in_channels // attention_head_dim * mult) if num_attention_heads is None else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim
        self.heads = num_attention_heads

        # Query, key, value projections
        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        # Multi-scale attention
        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(
                SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size)
            )

        # Output layers
        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, num_features=out_channels)

        # Set default processor
        self.fused_projections = False
        self.set_processor(self.default_processor_class())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process linear attention for Sana model inputs."""
        return self.processor(self, hidden_states)


class SanaMultiscaleAttentionProjection(nn.Module):
    """Projection layer for Sana multi-scale attention."""

    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        channels = 3 * in_channels
        self.proj_in = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states


@maybe_allow_in_graph
class SD3Attention(nn.Module, AttentionModuleMixin):
    """
    Attention implementation specialized for SD3 models.

    This module implements the joint attention mechanism used in SD3,
    with native support for context pre-processing.

    Args:
        query_dim (`int`): Number of channels in query.
        cross_attention_dim (`int`, *optional*): Number of channels in encoder states.
        heads (`int`, defaults to 8): Number of attention heads.
        dim_head (`int`, defaults to 64): Dimension of each attention head.
        dropout (`float`, defaults to 0.0): Dropout probability.
        bias (`bool`, defaults to False): Whether to use bias in linear projections.
        added_kv_proj_dim (`int`, *optional*): Dimension for added key/value projections.
    """

    # Set SD3-specific processor classes
    default_processor_class = JointAttnProcessorSDPA
    fused_processor_class = FusedJointAttnProcessorSDPA

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        context_pre_only: bool = False,
    ):
        super().__init__()

        # Core parameters
        self.inner_dim = dim_head * heads
        self.query_dim = query_dim
        self.heads = heads
        self.scale = dim_head**-0.5
        self.use_bias = bias
        self.scale_qk = True
        self.context_pre_only = context_pre_only

        # Cross-attention setup
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        # Projections for self-attention
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, self.inner_dim, bias=bias)

        # Added projections for context processing
        self.added_kv_proj_dim = added_kv_proj_dim
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=bias)
            self.added_proj_bias = bias

        # Output projection
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, query_dim, bias=bias), nn.Dropout(dropout)])

        # Context output projection
        if added_kv_proj_dim is not None and not context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=bias)
        else:
            self.to_add_out = None

        # Set default processor and fusion state
        self.fused_projections = False
        self.set_processor(self.default_processor_class())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Process joint attention for SD3 model inputs."""
        # Filter parameters to only those expected by the processor
        processor_params = inspect.signature(self.processor.__call__).parameters.keys()
        quiet_params = {"ip_adapter_masks", "ip_hidden_states"}

        # Check for unexpected parameters
        unexpected_params = [k for k, _ in kwargs.items() if k not in processor_params and k not in quiet_params]
        if unexpected_params:
            logger.warning(
                f"Parameters {unexpected_params} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )

        # Filter to only expected parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in processor_params}

        # Process with appropriate processor
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **filtered_kwargs,
        )

