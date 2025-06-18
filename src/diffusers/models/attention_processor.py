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
import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from ..image_processor import IPAdapterMaskProcessor
from ..utils import deprecate, is_torch_xla_available, logging
from ..utils.import_utils import is_torch_npu_available, is_torch_xla_version, is_xformers_available
from .attention_dispatch import dispatch_attention_fn


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_npu_available():
    pass

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

if is_torch_xla_available():
    # flash attention pallas kernel is introduced in the torch_xla 2.3 release.
    if is_torch_xla_version(">", "2.2"):
        from torch_xla.runtime import is_spmd
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class AttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        elementwise_affine: bool = True,
        is_causal: bool = False,
    ):
        super().__init__()

        # To prevent circular import.
        from .normalization import FP32LayerNorm, LpNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applies qk norm across all heads
            self.norm_q = nn.LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            self.norm_k = RMSNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
        elif qk_norm == "rms_norm_across_heads":
            # LTX applies qk norm across all heads
            self.norm_q = RMSNorm(dim_head * heads, eps=eps)
            self.norm_k = RMSNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "l2":
            self.norm_q = LpNorm(p=2, dim=-1, eps=eps)
            self.norm_k = LpNorm(p=2, dim=-1, eps=eps)
        else:
            raise ValueError(
                f"unknown qk_norm: {qk_norm}. Should be one of None, 'layer_norm', 'fp32_layer_norm', 'layer_norm_across_heads', 'rms_norm', 'rms_norm_across_heads', 'l2'."
            )

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_kv_dim, bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=added_proj_bias)
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
            self.to_out.append(nn.Dropout(dropout))
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=out_bias)
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "layer_norm":
                self.norm_added_q = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
                self.norm_added_k = nn.LayerNorm(dim_head, eps=eps, elementwise_affine=elementwise_affine)
            elif qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            elif qk_norm == "rms_norm_across_heads":
                # Wan applies qk norm across all heads
                # Wan also doesn't apply a q norm
                self.norm_added_q = None
                self.norm_added_k = RMSNorm(dim_head * kv_heads, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_xla_flash_attention(
        self,
        use_xla_flash_attention: bool,
        partition_spec: Optional[Tuple[Optional[str], ...]] = None,
        is_flux=False,
    ) -> None:
        r"""
        Set whether to use xla flash attention from `torch_xla` or not.

        Args:
            use_xla_flash_attention (`bool`):
                Whether to use pallas flash attention kernel from `torch_xla` or not.
            partition_spec (`Tuple[]`, *optional*):
                Specify the partition specification if using SPMD. Otherwise None.
        """
        if use_xla_flash_attention:
            if not is_torch_xla_available:
                raise "torch_xla is not available"
            elif is_torch_xla_version("<", "2.3"):
                raise "flash attention pallas kernel is supported from torch_xla version 2.3"
            elif is_spmd() and is_torch_xla_version("<", "2.4"):
                raise "flash attention pallas kernel using SPMD is supported from torch_xla version 2.4"
            else:
                if is_flux:
                    processor = XLAFluxFlashAttnProcessor2_0(partition_spec)
                else:
                    processor = XLAFlashAttnProcessor2_0(partition_spec)
        else:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
            ),
        )
        is_ip_adapter = hasattr(self, "processor") and isinstance(
            self.processor,
            (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0, IPAdapterXFormersAttnProcessor),
        )
        is_joint_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                JointAttnProcessor2_0,
                XFormersJointAttnProcessor,
            ),
        )

        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and is_custom_diffusion:
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    dtype = None
                    if attention_op is not None:
                        op_fw, op_bw = attention_op
                        dtype, *_ = op_fw.SUPPORTED_DTYPES
                    q = torch.randn((1, 2, 40), device="cuda", dtype=dtype)
                    _ = xformers.ops.memory_efficient_attention(q, q, q)
                except Exception as e:
                    raise e

            if is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            elif is_ip_adapter:
                processor = IPAdapterXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            elif is_joint_processor:
                processor = XFormersJointAttnProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomDiffusionAttnProcessor
                )
                processor = attn_processor_class(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_ip_adapter:
                processor = IPAdapterAttnProcessor2_0(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    num_tokens=self.processor.num_tokens,
                    scale=self.processor.scale,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_ip"):
                    processor.to(
                        device=self.processor.to_k_ip[0].weight.device, dtype=self.processor.to_k_ip[0].weight.dtype
                    )
            else:
                # set attention processor
                # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
                # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
                processor = (
                    AttnProcessor2_0()
                    if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                    else AttnProcessor()
                )

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(
                    head_size, dim=0, output_size=attention_mask.shape[0] * head_size
                )
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(
                head_size, dim=1, output_size=attention_mask.shape[1] * head_size
            )

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        # handle added projections for SD3 and others.
        if (
            getattr(self, "add_q_proj", None) is not None
            and getattr(self, "add_k_proj", None) is not None
            and getattr(self, "add_v_proj", None) is not None
        ):
            concatenated_weights = torch.cat(
                [self.add_q_proj.weight.data, self.add_k_proj.weight.data, self.add_v_proj.weight.data]
            )
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_added_qkv = nn.Linear(
                in_features, out_features, bias=self.added_proj_bias, device=device, dtype=dtype
            )
            self.to_added_qkv.weight.copy_(concatenated_weights)
            if self.added_proj_bias:
                concatenated_bias = torch.cat(
                    [self.add_q_proj.bias.data, self.add_k_proj.bias.data, self.add_v_proj.bias.data]
                )
                self.to_added_qkv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse


class SanaMultiscaleAttentionProjection(nn.Module):
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


class SanaMultiscaleLinearAttention(nn.Module):
    r"""Lightweight multi-scale linear attention"""

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

        # To prevent circular import
        from .normalization import get_normalization

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (
            int(in_channels // attention_head_dim * mult) if num_attention_heads is None else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(
                SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size)
            )

        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, num_features=out_channels)

        self.processor = SanaMultiscaleAttnProcessor2_0()

    def apply_linear_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)  # Adds padding
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
        return hidden_states

    def apply_quadratic_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(key.transpose(-1, -2), query)
        scores = scores.to(dtype=torch.float32)
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        hidden_states = torch.matmul(value, scores.to(value.dtype))
        return hidden_states

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)


class AttnAddedKVProcessor:
    r"""
    Processor for performing scaled dot-product attention (enabled by default if you're using PyTorch 2.0), with extra
    learnable key and value matrices for the text encoder.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnAddedKVProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query, out_dim=4)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj, out_dim=4)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj, out_dim=4)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key, out_dim=4)
            value = attn.head_to_batch_dim(value, out_dim=4)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, residual.shape[1])

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class JointAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("JointAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class PAGJointAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PAGJointAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # store the length of image patch sequences to create a mask that prevents interaction between patches
        # similar to making the self-attention map an identity matrix
        identity_block_size = hidden_states.shape[1]

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)
        encoder_hidden_states_org, encoder_hidden_states_ptb = encoder_hidden_states.chunk(2)

        ################## original path ##################
        batch_size = encoder_hidden_states_org.shape[0]

        # `sample` projections.
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        # `context` projections.
        encoder_hidden_states_org_query_proj = attn.add_q_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_key_proj = attn.add_k_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_value_proj = attn.add_v_proj(encoder_hidden_states_org)

        # attention
        query_org = torch.cat([query_org, encoder_hidden_states_org_query_proj], dim=1)
        key_org = torch.cat([key_org, encoder_hidden_states_org_key_proj], dim=1)
        value_org = torch.cat([value_org, encoder_hidden_states_org_value_proj], dim=1)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # Split the attention outputs.
        hidden_states_org, encoder_hidden_states_org = (
            hidden_states_org[:, : residual.shape[1]],
            hidden_states_org[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)
        if not attn.context_pre_only:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################## perturbed path ##################

        batch_size = encoder_hidden_states_ptb.shape[0]

        # `sample` projections.
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)

        # `context` projections.
        encoder_hidden_states_ptb_query_proj = attn.add_q_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_key_proj = attn.add_k_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_value_proj = attn.add_v_proj(encoder_hidden_states_ptb)

        # attention
        query_ptb = torch.cat([query_ptb, encoder_hidden_states_ptb_query_proj], dim=1)
        key_ptb = torch.cat([key_ptb, encoder_hidden_states_ptb_key_proj], dim=1)
        value_ptb = torch.cat([value_ptb, encoder_hidden_states_ptb_value_proj], dim=1)

        inner_dim = key_ptb.shape[-1]
        head_dim = inner_dim // attn.heads
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # create a full mask with all entries set to 0
        seq_len = query_ptb.size(2)
        full_mask = torch.zeros((seq_len, seq_len), device=query_ptb.device, dtype=query_ptb.dtype)

        # set the attention value between image patches to -inf
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")

        # set the diagonal of the attention value between image patches to 0
        full_mask[:identity_block_size, :identity_block_size].fill_diagonal_(0)

        # expand the mask to match the attention weights shape
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

        hidden_states_ptb = F.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

        # split the attention outputs.
        hidden_states_ptb, encoder_hidden_states_ptb = (
            hidden_states_ptb[:, : residual.shape[1]],
            hidden_states_ptb[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        if not attn.context_pre_only:
            encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################ concat ###############
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])

        return hidden_states, encoder_hidden_states


class PAGCFGJointAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGCFGJointAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        identity_block_size = hidden_states.shape[
            1
        ]  # patch embeddings width * height (correspond to self-attention map width or height)

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])

        (
            encoder_hidden_states_uncond,
            encoder_hidden_states_org,
            encoder_hidden_states_ptb,
        ) = encoder_hidden_states.chunk(3)
        encoder_hidden_states_org = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_org])

        ################## original path ##################
        batch_size = encoder_hidden_states_org.shape[0]

        # `sample` projections.
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        # `context` projections.
        encoder_hidden_states_org_query_proj = attn.add_q_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_key_proj = attn.add_k_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_value_proj = attn.add_v_proj(encoder_hidden_states_org)

        # attention
        query_org = torch.cat([query_org, encoder_hidden_states_org_query_proj], dim=1)
        key_org = torch.cat([key_org, encoder_hidden_states_org_key_proj], dim=1)
        value_org = torch.cat([value_org, encoder_hidden_states_org_value_proj], dim=1)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # Split the attention outputs.
        hidden_states_org, encoder_hidden_states_org = (
            hidden_states_org[:, : residual.shape[1]],
            hidden_states_org[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)
        if not attn.context_pre_only:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################## perturbed path ##################

        batch_size = encoder_hidden_states_ptb.shape[0]

        # `sample` projections.
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)

        # `context` projections.
        encoder_hidden_states_ptb_query_proj = attn.add_q_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_key_proj = attn.add_k_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_value_proj = attn.add_v_proj(encoder_hidden_states_ptb)

        # attention
        query_ptb = torch.cat([query_ptb, encoder_hidden_states_ptb_query_proj], dim=1)
        key_ptb = torch.cat([key_ptb, encoder_hidden_states_ptb_key_proj], dim=1)
        value_ptb = torch.cat([value_ptb, encoder_hidden_states_ptb_value_proj], dim=1)

        inner_dim = key_ptb.shape[-1]
        head_dim = inner_dim // attn.heads
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # create a full mask with all entries set to 0
        seq_len = query_ptb.size(2)
        full_mask = torch.zeros((seq_len, seq_len), device=query_ptb.device, dtype=query_ptb.dtype)

        # set the attention value between image patches to -inf
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")

        # set the diagonal of the attention value between image patches to 0
        full_mask[:identity_block_size, :identity_block_size].fill_diagonal_(0)

        # expand the mask to match the attention weights shape
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

        hidden_states_ptb = F.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

        # split the attention outputs.
        hidden_states_ptb, encoder_hidden_states_ptb = (
            hidden_states_ptb[:, : residual.shape[1]],
            hidden_states_ptb[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        if not attn.context_pre_only:
            encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################ concat ###############
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = torch.cat([encoder_hidden_states_org, encoder_hidden_states_ptb])

        return hidden_states, encoder_hidden_states


class XFormersJointAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = attn.head_to_batch_dim(encoder_hidden_states_query_proj).contiguous()
            encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj).contiguous()
            encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj).contiguous()

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class MochiVaeAttnProcessor:
    r"""
    Attention processor used in Mochi VAE.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        is_single_frame = hidden_states.shape[1] == 1

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if is_single_frame:
            hidden_states = attn.to_v(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            return hidden_states

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=attn.is_causal,
            backend=self._attention_backend,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class StableAudioAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the Stable Audio model. It applies rotary embedding on query and key vector, and allows MHA, GQA or MQA.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "StableAudioAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def apply_partial_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        rot_dim = freqs_cis[0].shape[-1]
        x_to_rotate, x_unrotated = x[..., :rot_dim], x[..., rot_dim:]

        x_rotated = apply_rotary_emb(x_to_rotate, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        out = torch.cat((x_rotated, x_unrotated), dim=-1)
        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if kv_heads != attn.heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = attn.heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1, output_size=key.shape[1] * heads_per_kv_head)
            value = torch.repeat_interleave(
                value, heads_per_kv_head, dim=1, output_size=value.shape[1] * heads_per_kv_head
            )

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_emb is not None:
            query_dtype = query.dtype
            key_dtype = key.dtype
            query = query.to(torch.float32)
            key = key.to(torch.float32)

            rot_dim = rotary_emb[0].shape[-1]
            query_to_rotate, query_unrotated = query[..., :rot_dim], query[..., rot_dim:]
            query_rotated = apply_rotary_emb(query_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

            query = torch.cat((query_rotated, query_unrotated), dim=-1)

            if not attn.is_cross_attention:
                key_to_rotate, key_unrotated = key[..., :rot_dim], key[..., rot_dim:]
                key_rotated = apply_rotary_emb(key_to_rotate, rotary_emb, use_real=True, use_real_unbind_dim=-2)

                key = torch.cat((key_rotated, key_unrotated), dim=-1)

            query = query.to(query_dtype)
            key = key.to(key_dtype)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class FusedHunyuanAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0) with fused
    projection layers. This is used in the HunyuanDiT model. It applies a s normalization layer and rotary embedding on
    query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedHunyuanAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // 2
            key, value = torch.split(kv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PAGHunyuanAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a normalization layer and rotary embedding on query and key vector. This
    variant of the processor employs [Pertubed Attention Guidance](https://huggingface.co/papers/2403.17377).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGHunyuanAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        # 1. Original Path
        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states_org)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states_org
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 2. Perturbed Path
        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PAGCFGHunyuanAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the HunyuanDiT model. It applies a normalization layer and rotary embedding on query and key vector. This
    variant of the processor employs [Pertubed Attention Guidance](https://huggingface.co/papers/2403.17377).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGCFGHunyuanAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])

        # 1. Original Path
        batch_size, sequence_length, _ = (
            hidden_states_org.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states_org)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states_org
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # 2. Perturbed Path
        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class LuminaAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). This is
    used in the LuminaNextDiT model. It applies a s normalization layer and rotary embedding on query and key vector.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb: Optional[torch.Tensor] = None,
        key_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        from .embeddings import apply_rotary_emb

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)

        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        return hidden_states


class FusedAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0). It uses
    fused projection layers. For self-attention modules, all projection matrices (i.e., query, key, value) are fused.
    For cross-attention modules, key and value projection matrices are fused.

    <Tip warning={true}>

    This API is currently 🧪 experimental in nature and can change in future.

    </Tip>
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FusedAttnProcessor requires at least PyTorch 2.0, to use it. Please upgrade PyTorch to > 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            qkv = attn.to_qkv(hidden_states)
            split_size = qkv.shape[-1] // 3
            query, key, value = torch.split(qkv, split_size, dim=-1)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            query = attn.to_q(hidden_states)

            kv = attn.to_kv(encoder_hidden_states)
            split_size = kv.shape[-1] // 2
            key, value = torch.split(kv, split_size, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomDiffusionXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers for the Custom Diffusion method.

    Args:
    train_kv (`bool`, defaults to `True`):
        Whether to newly train the key and value matrices corresponding to the text features.
    train_q_out (`bool`, defaults to `True`):
        Whether to newly train query matrices corresponding to the latent image features.
    hidden_size (`int`, *optional*, defaults to `None`):
        The hidden size of the attention layer.
    cross_attention_dim (`int`, *optional*, defaults to `None`):
        The number of channels in the `encoder_hidden_states`.
    out_bias (`bool`, defaults to `True`):
        Whether to include the bias parameter in `train_q_out`.
    dropout (`float`, *optional*, defaults to 0.0):
        The dropout probability to use.
    attention_op (`Callable`, *optional*, defaults to `None`):
        The base
        [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to use
        as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best operator.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = False,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
        attention_op: Optional[Callable] = None,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states).to(attn.to_q.weight.dtype)
        else:
            query = attn.to_q(hidden_states.to(attn.to_q.weight.dtype))

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)
        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CustomDiffusionAttnProcessor(nn.Module):
    r"""
    Processor for implementing attention for the Custom Diffusion method using PyTorch 2.0’s memory-efficient scaled
    dot-product attention.

    Args:
        train_kv (`bool`, defaults to `True`):
            Whether to newly train the key and value matrices corresponding to the text features.
        train_q_out (`bool`, defaults to `True`):
            Whether to newly train query matrices corresponding to the latent image features.
        hidden_size (`int`, *optional*, defaults to `None`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*, defaults to `None`):
            The number of channels in the `encoder_hidden_states`.
        out_bias (`bool`, defaults to `True`):
            Whether to include the bias parameter in `train_q_out`.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
    """

    def __init__(
        self,
        train_kv: bool = True,
        train_q_out: bool = True,
        hidden_size: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,
        out_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.train_kv = train_kv
        self.train_q_out = train_q_out

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        # `_custom_diffusion` id for easy serialization and loading.
        if self.train_kv:
            self.to_k_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
            self.to_v_custom_diffusion = nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False)
        if self.train_q_out:
            self.to_q_custom_diffusion = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_custom_diffusion = nn.ModuleList([])
            self.to_out_custom_diffusion.append(nn.Linear(hidden_size, hidden_size, bias=out_bias))
            self.to_out_custom_diffusion.append(nn.Dropout(dropout))

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if self.train_q_out:
            query = self.to_q_custom_diffusion(hidden_states)
        else:
            query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            crossattn = False
            encoder_hidden_states = hidden_states
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        if self.train_kv:
            key = self.to_k_custom_diffusion(encoder_hidden_states.to(self.to_k_custom_diffusion.weight.dtype))
            value = self.to_v_custom_diffusion(encoder_hidden_states.to(self.to_v_custom_diffusion.weight.dtype))
            key = key.to(attn.to_q.weight.dtype)
            value = value.to(attn.to_q.weight.dtype)

        else:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)

        if crossattn:
            detach = torch.ones_like(key)
            detach[:, :1, :] = detach[:, :1, :] * 0.0
            key = detach * key + (1 - detach) * key.detach()
            value = detach * value + (1 - detach) * value.detach()

        inner_dim = hidden_states.shape[-1]

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if self.train_q_out:
            # linear proj
            hidden_states = self.to_out_custom_diffusion[0](hidden_states)
            # dropout
            hidden_states = self.to_out_custom_diffusion[1](hidden_states)
        else:
            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnProcessor:
    r"""
    Processor for implementing sliced attention.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size: int):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SlicedAttnAddedKVProcessor:
    r"""
    Processor for implementing sliced attention with extra learnable key and value matrices for the text encoder.

    Args:
        slice_size (`int`, *optional*):
            The number of steps to compute attention. Uses as many slices as `attention_head_dim // slice_size`, and
            `attention_head_dim` must be a multiple of the `slice_size`.
    """

    def __init__(self, slice_size):
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_key_proj = attn.head_to_batch_dim(encoder_hidden_states_key_proj)
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(encoder_hidden_states_value_proj)

        if not attn.only_cross_attention:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
        else:
            key = encoder_hidden_states_key_proj
            value = encoder_hidden_states_value_proj

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads), device=query.device, dtype=query.dtype
        )

        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://huggingface.co/papers/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_channels=f_channels, num_groups=32, eps=1e-6, affine=True)
        self.conv_y = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_b = nn.Conv2d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


class IPAdapterAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if mask is None:
                        continue
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        # the output of sdp = (batch, num_heads, seq_len, head_dim)
                        # TODO: add support for attn.scale when we move to Torch 2.1
                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    # the output of sdp = (batch, num_heads, seq_len, head_dim)
                    # TODO: add support for attn.scale when we move to Torch 2.1
                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class IPAdapterXFormersAttnProcessor(torch.nn.Module):
    r"""
    Attention processor for IP-Adapter using xFormers.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        num_tokens=(4,),
        scale=1.0,
        attention_op: Optional[Callable] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.attention_op = attention_op

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim or hidden_size, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if ip_hidden_states:
            if ip_adapter_masks is not None:
                if not isinstance(ip_adapter_masks, List):
                    # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                    ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
                if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                    raise ValueError(
                        f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                        f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                        f"({len(ip_hidden_states)})"
                    )
                else:
                    for index, (mask, scale, ip_state) in enumerate(
                        zip(ip_adapter_masks, self.scale, ip_hidden_states)
                    ):
                        if mask is None:
                            continue
                        if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                            raise ValueError(
                                "Each element of the ip_adapter_masks array should be a tensor with shape "
                                "[1, num_images_for_ip_adapter, height, width]."
                                " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                            )
                        if mask.shape[1] != ip_state.shape[1]:
                            raise ValueError(
                                f"Number of masks ({mask.shape[1]}) does not match "
                                f"number of ip images ({ip_state.shape[1]}) at index {index}"
                            )
                        if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                            raise ValueError(
                                f"Number of masks ({mask.shape[1]}) does not match "
                                f"number of scales ({len(scale)}) at index {index}"
                            )
            else:
                ip_adapter_masks = [None] * len(self.scale)

            # for ip-adapter
            for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
                ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
            ):
                skip = False
                if isinstance(scale, list):
                    if all(s == 0 for s in scale):
                        skip = True
                elif scale == 0:
                    skip = True
                if not skip:
                    if mask is not None:
                        mask = mask.to(torch.float16)
                        if not isinstance(scale, list):
                            scale = [scale] * mask.shape[1]

                        current_num_images = mask.shape[1]
                        for i in range(current_num_images):
                            ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                            ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                            ip_key = attn.head_to_batch_dim(ip_key).contiguous()
                            ip_value = attn.head_to_batch_dim(ip_value).contiguous()

                            _current_ip_hidden_states = xformers.ops.memory_efficient_attention(
                                query, ip_key, ip_value, op=self.attention_op
                            )
                            _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)
                            _current_ip_hidden_states = attn.batch_to_head_dim(_current_ip_hidden_states)

                            mask_downsample = IPAdapterMaskProcessor.downsample(
                                mask[:, i, :, :],
                                batch_size,
                                _current_ip_hidden_states.shape[1],
                                _current_ip_hidden_states.shape[2],
                            )

                            mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                            hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
                    else:
                        ip_key = to_k_ip(current_ip_hidden_states)
                        ip_value = to_v_ip(current_ip_hidden_states)

                        ip_key = attn.head_to_batch_dim(ip_key).contiguous()
                        ip_value = attn.head_to_batch_dim(ip_value).contiguous()

                        current_ip_hidden_states = xformers.ops.memory_efficient_attention(
                            query, ip_key, ip_value, op=self.attention_op
                        )
                        current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
                        current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

                        hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SD3IPAdapterJointAttnProcessor(torch.nn.Module):
    """
    Attention processor for IP-Adapter used typically in processing the SD3-like self-attention projections, with
    additional image-based information and timestep embeddings.

    Args:
        hidden_size (`int`):
            The number of hidden channels.
        ip_hidden_states_dim (`int`):
            The image feature dimension.
        head_dim (`int`):
            The number of head channels.
        timesteps_emb_dim (`int`, defaults to 1280):
            The number of input channels for timestep embedding.
        scale (`float`, defaults to 0.5):
            IP-Adapter scale.
    """

    def __init__(
        self,
        hidden_size: int,
        ip_hidden_states_dim: int,
        head_dim: int,
        timesteps_emb_dim: int = 1280,
        scale: float = 0.5,
    ):
        super().__init__()

        # To prevent circular import
        from .normalization import AdaLayerNorm, RMSNorm

        self.norm_ip = AdaLayerNorm(timesteps_emb_dim, output_dim=ip_hidden_states_dim * 2, norm_eps=1e-6, chunk_dim=1)
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)
        self.scale = scale

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        ip_hidden_states: torch.FloatTensor = None,
        temb: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Perform the attention computation, integrating image features (if provided) and timestep embeddings.

        If `ip_hidden_states` is `None`, this is equivalent to using JointAttnProcessor2_0.

        Args:
            attn (`Attention`):
                Attention instance.
            hidden_states (`torch.FloatTensor`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor`, *optional*):
                The encoder hidden states.
            attention_mask (`torch.FloatTensor`, *optional*):
                Attention mask.
            ip_hidden_states (`torch.FloatTensor`, *optional*):
                Image embeddings.
            temb (`torch.FloatTensor`, *optional*):
                Timestep embeddings.

        Returns:
            `torch.FloatTensor`: Output hidden states.
        """
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        img_query = query
        img_key = key
        img_value = value

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # IP Adapter
        if self.scale != 0 and ip_hidden_states is not None:
            # Norm image features
            norm_ip_hidden_states = self.norm_ip(ip_hidden_states, temb=temb)

            # To k and v
            ip_key = self.to_k_ip(norm_ip_hidden_states)
            ip_value = self.to_v_ip(norm_ip_hidden_states)

            # Reshape
            ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            # Norm
            query = self.norm_q(img_query)
            img_key = self.norm_k(img_key)
            ip_key = self.norm_ip_k(ip_key)

            # cat img
            key = torch.cat([img_key, ip_key], dim=2)
            value = torch.cat([img_value, ip_value], dim=2)

            ip_hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            ip_hidden_states = ip_hidden_states.transpose(1, 2).view(batch_size, -1, attn.heads * head_dim)
            ip_hidden_states = ip_hidden_states.to(query.dtype)

            hidden_states = hidden_states + ip_hidden_states * self.scale

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class PAGIdentitySelfAttnProcessor:
    r"""
    Processor for implementing PAG using scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    PAG reference: https://huggingface.co/papers/2403.17377
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGIdentitySelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # chunk
        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        # original path
        batch_size, sequence_length, _ = hidden_states_org.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # perturbed path (identity attention)
        batch_size, sequence_length, _ = hidden_states_ptb.shape

        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

        hidden_states_ptb = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class PAGCFGIdentitySelfAttnProcessor:
    r"""
    Processor for implementing PAG using scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    PAG reference: https://huggingface.co/papers/2403.17377
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "PAGCFGIdentitySelfAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])

        # original path
        batch_size, sequence_length, _ = hidden_states_org.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states_org = attn.group_norm(hidden_states_org.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states_org = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query.dtype)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # perturbed path (identity attention)
        batch_size, sequence_length, _ = hidden_states_ptb.shape

        if attn.group_norm is not None:
            hidden_states_ptb = attn.group_norm(hidden_states_ptb.transpose(1, 2)).transpose(1, 2)

        value = attn.to_v(hidden_states_ptb)
        hidden_states_ptb = value
        hidden_states_ptb = hidden_states_ptb.to(query.dtype)

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # cat
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SanaMultiscaleAttnProcessor:
    r"""
    Processor for implementing multiscale quadratic attention.
    """

    def __call__(self, attn: SanaMultiscaleLinearAttention, hidden_states: torch.Tensor) -> torch.Tensor:
        height, width = hidden_states.shape[-2:]
        if height * width > attn.attention_head_dim:
            use_linear_attention = True
        else:
            use_linear_attention = False

        residual = hidden_states

        batch_size, _, height, width = list(hidden_states.size())
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in attn.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))

        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            # for linear attention upcast hidden_states to float32
            hidden_states = hidden_states.to(dtype=torch.float32)

        hidden_states = hidden_states.reshape(batch_size, -1, 3 * attn.attention_head_dim, height * width)

        query, key, value = hidden_states.chunk(3, dim=2)
        query = attn.nonlinearity(query)
        key = attn.nonlinearity(key)

        if use_linear_attention:
            hidden_states = attn.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = attn.apply_quadratic_attention(query, key, value)

        hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = attn.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if attn.norm_type == "rms_norm":
            hidden_states = attn.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = attn.norm_out(hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class LoRAAttnProcessor:
    r"""
    Processor for implementing attention with LoRA.
    """

    def __init__(self):
        pass


class LoRAAttnProcessor:
    r"""
    Processor for implementing attention with LoRA (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        pass


class LoRAXFormersAttnProcessor:
    r"""
    Processor for implementing attention with LoRA using xFormers.
    """

    def __init__(self):
        pass


class LoRAAttnAddedKVProcessor:
    r"""
    Processor for implementing attention with LoRA with extra learnable key and value matrices for the text encoder.
    """

    def __init__(self):
        pass


class SanaLinearAttnProcessor:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + 1e-15)
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class PAGCFGSanaLinearAttnProcessor:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states_org = torch.matmul(scores, query)

        hidden_states_org = hidden_states_org[:, :, :-1] / (hidden_states_org[:, :, -1:] + 1e-15)
        hidden_states_org = hidden_states_org.flatten(1, 2).transpose(1, 2)
        hidden_states_org = hidden_states_org.to(original_dtype)

        hidden_states_org = attn.to_out[0](hidden_states_org)
        hidden_states_org = attn.to_out[1](hidden_states_org)

        # perturbed path (identity attention)
        hidden_states_ptb = attn.to_v(hidden_states_ptb).to(original_dtype)

        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class PAGIdentitySanaLinearAttnProcessor:
    r"""
    Processor for implementing scaled dot-product linear attention.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype

        hidden_states_org, hidden_states_ptb = hidden_states.chunk(2)

        query = attn.to_q(hidden_states_org)
        key = attn.to_k(hidden_states_org)
        value = attn.to_v(hidden_states_org)

        query = query.transpose(1, 2).unflatten(1, (attn.heads, -1))
        key = key.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)
        value = value.transpose(1, 2).unflatten(1, (attn.heads, -1))

        query = F.relu(query)
        key = F.relu(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1.0)
        scores = torch.matmul(value, key)
        hidden_states_org = torch.matmul(scores, query)

        if hidden_states_org.dtype in [torch.float16, torch.bfloat16]:
            hidden_states_org = hidden_states_org.float()

        hidden_states_org = hidden_states_org[:, :, :-1] / (hidden_states_org[:, :, -1:] + 1e-15)
        hidden_states_org = hidden_states_org.flatten(1, 2).transpose(1, 2)
        hidden_states_org = hidden_states_org.to(original_dtype)

        hidden_states_org = attn.to_out[0](hidden_states_org)
        hidden_states_org = attn.to_out[1](hidden_states_org)

        # perturbed path (identity attention)
        hidden_states_ptb = attn.to_v(hidden_states_ptb).to(original_dtype)

        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)

        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])

        if original_dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


# Deprecated classes for backward compatibility


class AttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "`AttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `AttnProcessor`"
        )
        deprecate("AttnProcessor2_0", "1.0.0", deprecation_message)

        return AttnProcessor(*args, **kwargs)


class AttnAddedKVProcessor:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`AttnAddedKVAttentionProcessor` is deprecated and this will be removed in a future version. Please use `AttnAddedKVProcessor`"
        deprecate("AttnAddedKVAttentionProcessor", "1.0.0", deprecation_message)

        return AttnAddedKVProcessor(*args, **kwargs)


class AttnAddedKVProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`AttnAddedKVAttentionProcessor` is deprecated and this will be removed in a future version. Please use `AttnAddedKVProcessor`"
        deprecate("AttnAddedKVAttentionProcessor", "1.0.0", deprecation_message)

        return AttnAddedKVProcessor(*args, **kwargs)


class AllegroAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`AllegroAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `AllegroAttnProcessor`"
        deprecate("AllegroAttnProcessor2_0", "1.0.0", deprecation_message)

        return AllegroAttnProcessor(*args, **kwargs)


class AuraFlowAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`AuraFlowAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `AuraFlowAttnProcessor`"
        deprecate("AuraFlowAttnProcessor2_0", "1.0.0", deprecation_message)

        return AuraFlowAttnProcessor(*args, **kwargs)


class MochiAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`MochiAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `MochiAttnProcessor`"
        deprecate("MochiAttnProcessor2_0", "1.0.0", deprecation_message)

        from .transformers.transformer_mochi import MochiAttnProcessor

        return MochiAttnProcessor(*args, **kwargs)


class MochiVaeAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`MochiVaeAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `MochiVaeAttnProcessor`"
        deprecate("MochiVaeAttnProcessor2_0", "1.0.0", deprecation_message)

        from .autoencoders.autoencoder_kl_mochi import MochiVaeAttnProcessor

        return MochiVaeAttnProcessor(*args, **kwargs)


class FluxAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FluxAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `FluxAttnProcessor`"
        deprecate("FluxAttnProcessor2_0", "1.0.0", deprecation_message)

        from .transformers.transformer_flux import FluxAttnProcessor

        return FluxAttnProcessor(*args, **kwargs)


class FluxSingleAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FluxSingleAttnProcessor` is deprecated and will be removed in a future version. Please use `FluxAttnProcessorSDPA` instead."
        deprecate("FluxSingleAttnProcessor2_0", "1.0.0", deprecation_message)

        from .transformers.transformer_flux import FluxAttnProcessor

        return FluxAttnProcessor(*args, **kwargs)


class FusedAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `AttnProcessor`"
        deprecate("FusedAttnProcessor2_0", "1.0.0", deprecation_message)

        return AttnProcessor(*args, **kwargs)


class JointAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`JointAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `JointAttnProcessor`"
        deprecate("JointAttnProcessor2_0", "1.0.0", deprecation_message)

        return JointAttnProcessor(*args, **kwargs)


class PAGJointAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGJointAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGJointAttnProcessor`"
        deprecate("PAGJointAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGJointAttnProcessor(*args, **kwargs)


class PAGCFGJointAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGCFGJointAttnProcessor2_0 is deprecated and this will be removed in a future version. Please use `PAGCFGJointAttnProcessor`"
        deprecate("PAGCFGJointAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGCFGJointAttnProcessor(*args, **kwargs)


class FusedJointAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedJointAttnProcessor2_0 is deprecated and this will be removed in a future version. Please use `JointAttnProcessor`"
        deprecate("FusedJointAttnProcessor2_0", "1.0.0", deprecation_message)

        return JointAttnProcessor(*args, **kwargs)


class FusedAuraFlowAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedAuraFlowAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `AuraFlowAttnProcessor`"
        deprecate("FusedAuraFlowAttnProcessor2_0", "1.0.0", deprecation_message)

        return AuraFlowAttnProcessor(*args, **kwargs)


class FusedFluxAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedFluxAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `FluxAttnProcessor`"
        deprecate("FusedFluxAttnProcessor2_0", "1.0.0", deprecation_message)
        from .transformers.transformer_flux import FluxAttnProcessor

        return FluxAttnProcessor(*args, **kwargs)


class CogVideoXAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`CogVideoXAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `CogVideoXAttnProcessor`"
        deprecate("CogVideoXAttnProcessor2_0", "1.0.0", deprecation_message)
        from .transformers.cogvideox_transformer_3d import CogVideoXAttnProcessor

        return CogVideoXAttnProcessor(*args, **kwargs)


class FusedCogVideoXAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedCogVideoXAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `CogVideoXAttnProcessor`"
        deprecate("FusedCogVideoXAttnProcessor2_0", "1.0.0", deprecation_message)
        from .transformers.cogvideox_transformer_3d import CogVideoXAttnProcessor

        return CogVideoXAttnProcessor(*args, **kwargs)


class XLAFlashAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`XLAFlashAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `XLAFlashAttnProcessor`"
        deprecate("XLAFlashAttnProcessor2_0", "1.0.0", deprecation_message)

        return XLAFlashAttnProcessor(*args, **kwargs)


class XLAFluxFlashAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`XLAFluxFlashAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `XLAFluxFlashAttnProcessor`"
        deprecate("XLAFluxFlashAttnProcessor2_0", "1.0.0", deprecation_message)

        from transformers.transformer_flux import FluxAttnProcessorXLA

        return FluxAttnProcessorXLA(*args, **kwargs)


class StableAudioAttnProcessor2_0:
    def __new__(self, *args, **kwargs):
        deprecation_message = "`StableAudioAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `StableAudioAttnProcessor`"
        deprecate("StableAudioAttnProcessor2_0", "1.0.0", deprecation_message)

        return StableAudioAttnProcessor(*args, **kwargs)


class HunyuanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`HunyuanAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `HunyuanAttnProcessor`"
        deprecate("HunyuanAttnProcessor2_0", "1.0.0", deprecation_message)

        return HunyuanAttnProcessor(*args, **kwargs)


class FusedHunyuanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`FusedHunyuanAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `FusedHunyuanAttnProcessor`"
        deprecate("FusedHunyuanAttnProcessor2_0", "1.0.0", deprecation_message)

        return HunyuanAttnProcessor(*args, **kwargs)


class PAGHunyuanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGHunyuanAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGHunyuanAttnProcessor`"
        deprecate("PAGHunyuanAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGHunyuanAttnProcessor(*args, **kwargs)


class PAGCFGHunyuanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGCFGHunyuanAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGCFGHunyuanAttnProcessor`"
        deprecate("PAGCFGHunyuanAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGCFGHunyuanAttnProcessor(*args, **kwargs)


class LuminaAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`LuminaAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `LuminaAttnProcessor`"
        deprecate("LuminaAttnProcessor2_0", "1.0.0", deprecation_message)

        return LuminaAttnProcessor(*args, **kwargs)


class PAGIdentitySelfAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGIdentitySelfAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGIdentitySelfAttnProcessor`"
        deprecate("PAGIdentitySelfAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGIdentitySelfAttnProcessor(*args, **kwargs)


class PAGCFGIdentitySelfAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGCFGIdentitySelfAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGCFGIdentitySelfAttnProcessor`"
        deprecate("PAGCFGIdentitySelfAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGCFGIdentitySelfAttnProcessor(*args, **kwargs)


class SanaMultiscaleAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`SanaMultiscaleAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `SanaMultiscaleAttnProcessor`"
        deprecate("SanaMultiscaleAttnProcessor2_0", "1.0.0", deprecation_message)

        return SanaMultiscaleAttnProcessor(*args, **kwargs)


class LoRAAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`LoRAAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `LoRAAttnProcessor`"
        deprecate("LoRAAttnProcessor2_0", "1.0.0", deprecation_message)

        return LoRAAttnProcessor(*args, **kwargs)


class SanaLinearAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`SanaLinearAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `SanaLinearAttnProcessor`"
        deprecate("SanaLinearAttnProcessor2_0", "1.0.0", deprecation_message)

        return SanaLinearAttnProcessor(*args, **kwargs)


class PAGCFGSanaLinearAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGCFGSanaLinearAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGCFGSanaLinearAttnProcessor`"
        deprecate("PAGCFGSanaLinearAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGCFGSanaLinearAttnProcessor(*args, **kwargs)


class PAGIdentitySanaLinearAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`PAGIdentitySanaLinearAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `PAGIdentitySanaLinearAttnProcessor`"
        deprecate("PAGIdentitySanaLinearAttnProcessor2_0", "1.0.0", deprecation_message)

        return PAGIdentitySanaLinearAttnProcessor(*args, **kwargs)


class IPAdapterAttnProcessor(IPAdapterAttnProcessor):
    def __init__(self, *args, **kwargs):
        deprecation_message = "`IPAdapterAttnProcessor` is deprecated and this will be removed in a future version. Please use `IPAdapterAttnProcessor`"
        deprecate("IPAdapterAttnProcessor", "1.0.0", deprecation_message)
        super().__init__(*args, **kwargs)


class IPAdapterAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = "`IPAdapterAttnProcessor2_0` is deprecated and this will be removed in a future version. Please use `IPAdapterAttnProcessor`"
        deprecate("IPAdapterAttnProcessor2_0", "1.0.0", deprecation_message)

        return IPAdapterAttnProcessor(*args, **kwargs)


ADDED_KV_ATTENTION_PROCESSORS = (
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    AttnAddedKVProcessor,
    XFormersAttnAddedKVProcessor,
)

CROSS_ATTENTION_PROCESSORS = (
    AttnProcessor,
    AttnProcessor,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterAttnProcessor,
)

AttentionProcessor = Union[
    AttnAddedKVProcessor,
    JointAttnProcessor,
    PAGJointAttnProcessor,
    PAGCFGJointAttnProcessor,
    FusedJointAttnProcessor,
    FusedAuraFlowAttnProcessor,
    CogVideoXAttnProcessor,
    FusedCogVideoXAttnProcessor,
    XFormersAttnAddedKVProcessor,
    XFormersAttnProcessor,
    XLAFlashAttnProcessor,
    AttnProcessorNPU,
    AttnProcessor,
    MochiVaeAttnProcessor,
    StableAudioAttnProcessor,
    FusedHunyuanAttnProcessor,
    PAGHunyuanAttnProcessor,
    PAGCFGHunyuanAttnProcessor,
    LuminaAttnProcessor,
    FusedAttnProcessor,
    CustomDiffusionXFormersAttnProcessor,
    CustomDiffusionAttnProcessor,
    SlicedAttnProcessor,
    SlicedAttnAddedKVProcessor,
    SanaLinearAttnProcessor,
    PAGCFGSanaLinearAttnProcessor,
    PAGIdentitySanaLinearAttnProcessor,
    SanaMultiscaleAttnProcessor,
    IPAdapterAttnProcessor,
    IPAdapterXFormersAttnProcessor,
    SD3IPAdapterJointAttnProcessor,
    PAGIdentitySelfAttnProcessor,
    PAGCFGIdentitySelfAttnProcessor,
    LoRAAttnProcessor,
    LoRAXFormersAttnProcessor,
    LoRAAttnAddedKVProcessor,
]
