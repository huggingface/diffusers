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

from typing import Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import logging
from ..utils.import_utils import is_torch_npu_available, is_torch_xla_available, is_xformers_available
from .attention_processor import Attention, AttentionProcessor  # noqa


if is_xformers_available():
    import xformers as xops
else:
    xops = None


logger = logging.get_logger(__name__)


class AttentionMixin:
    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.
        """
        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        for module in self.modules():
            if isinstance(module, AttentionModuleMixin):
                module.fuse_projections()

    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        for module in self.modules():
            if isinstance(module, AttentionModuleMixin):
                module.unfuse_projections()


class AttentionModuleMixin:
    _default_processor_cls = None
    _available_processors = []
    fused_projections = False

    def set_processor(self, processor: AttentionProcessor) -> None:
        """
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
        """
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

    def set_attention_backend(self, backend: str):
        from .attention_dispatch import AttentionBackendName

        available_backends = {x.value for x in AttentionBackendName.__members__.values()}
        if backend not in available_backends:
            raise ValueError(f"`{backend=}` must be one of the following: " + ", ".join(available_backends))

        backend = AttentionBackendName(backend.lower())
        self.processor._attention_backend = backend

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        """
        Set whether to use NPU flash attention from `torch_npu` or not.

        Args:
            use_npu_flash_attention (`bool`): Whether to use NPU flash attention or not.
        """

        if use_npu_flash_attention:
            if not is_torch_npu_available():
                raise ImportError("torch_npu is not available")

        self.set_attention_backend("_native_npu")

    def set_use_xla_flash_attention(
        self,
        use_xla_flash_attention: bool,
        partition_spec: Optional[Tuple[Optional[str], ...]] = None,
        is_flux=False,
    ) -> None:
        """
        Set whether to use XLA flash attention from `torch_xla` or not.

        Args:
            use_xla_flash_attention (`bool`):
                Whether to use pallas flash attention kernel from `torch_xla` or not.
            partition_spec (`Tuple[]`, *optional*):
                Specify the partition specification if using SPMD. Otherwise None.
            is_flux (`bool`, *optional*, defaults to `False`):
                Whether the model is a Flux model.
        """
        if use_xla_flash_attention:
            if not is_torch_xla_available():
                raise ImportError("torch_xla is not available")

        self.set_attention_backend("_native_xla")

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        """
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    "Refer to https://github.com/facebookresearch/xformers for more information on how to install xformers",
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
                    if is_xformers_available():
                        dtype = None
                        if attention_op is not None:
                            op_fw, op_bw = attention_op
                            dtype, *_ = op_fw.SUPPORTED_DTYPES
                        q = torch.randn((1, 2, 40), device="cuda", dtype=dtype)
                        _ = xops.memory_efficient_attention(q, q, q)
                except Exception as e:
                    raise e

                self.set_attention_backend("xformers")

    @torch.no_grad()
    def fuse_projections(self):
        """
        Fuse the query, key, and value projections into a single projection for efficiency.
        """
        # Skip if already fused
        if getattr(self, "fused_projections", False):
            return

        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if hasattr(self, "is_cross_attention") and self.is_cross_attention:
            # Fuse cross-attention key-value projections
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if hasattr(self, "use_bias") and self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)
        else:
            # Fuse self-attention projections
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if hasattr(self, "use_bias") and self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        # Handle added projections for models like SD3, Flux, etc.
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

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        """
        Unfuse the query, key, and value projections back to separate projections.
        """
        # Skip if not fused
        if not getattr(self, "fused_projections", False):
            return

        # Remove fused projection layers
        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")

        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")

        if hasattr(self, "to_added_qkv"):
            delattr(self, "to_added_qkv")

        self.fused_projections = False

    def set_attention_slice(self, slice_size: int) -> None:
        """
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if hasattr(self, "sliceable_head_dim") and slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        processor = None

        # Try to get a compatible processor for sliced attention
        if slice_size is not None:
            processor = self._get_compatible_processor("sliced")

        # If no processor was found or slice_size is None, use default processor
        if processor is None:
            processor = self.default_processor_cls()

        self.set_processor(processor)

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`.

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
        """
        Reshape the tensor for multi-head attention processing.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor.

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
        """
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use.

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
        """
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`): The attention mask to prepare.
            target_length (`int`): The target length of the attention mask.
            batch_size (`int`): The batch size for repeating the attention mask.
            out_dim (`int`, *optional*, defaults to `3`): Output dimension.

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
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Normalize the encoder hidden states.

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


def _chunked_feed_forward(*args, **kwargs):
    """Backward compatibility stub. Use transformers.modeling_common._chunked_feed_forward instead."""
    logger.warning(
        "Importing `_chunked_feed_forward` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
        "Please use `from diffusers.models.transformers.modeling_common import _chunked_feed_forward` instead."
    )
    from .transformers.modeling_common import _chunked_feed_forward as _actual_chunked_feed_forward

    return _actual_chunked_feed_forward(*args, **kwargs)


class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()
        from .transformers.modeling_common import FeedForward

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


class JointTransformerBlock:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.JointTransformerBlock instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `JointTransformerBlock` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import JointTransformerBlock` instead."
        )
        from .transformers.modeling_common import JointTransformerBlock

        return JointTransformerBlock(*args, **kwargs)


class BasicTransformerBlock:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.BasicTransformerBlock instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `BasicTransformerBlock` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import BasicTransformerBlock` instead."
        )
        from .transformers.modeling_common import BasicTransformerBlock

        return BasicTransformerBlock(*args, **kwargs)


class LuminaFeedForward:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.LuminaFeedForward instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `LuminaFeedForward` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import LuminaFeedForward` instead."
        )
        from .transformers.modeling_common import LuminaFeedForward

        return LuminaFeedForward(*args, **kwargs)


class TemporalBasicTransformerBlock:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.TemporalBasicTransformerBlock instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `TemporalBasicTransformerBlock` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import TemporalBasicTransformerBlock` instead."
        )
        from .transformers.modeling_common import TemporalBasicTransformerBlock

        return TemporalBasicTransformerBlock(*args, **kwargs)


class SkipFFTransformerBlock:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.SkipFFTransformerBlock instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `SkipFFTransformerBlock` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import SkipFFTransformerBlock` instead."
        )
        from .transformers.modeling_common import SkipFFTransformerBlock

        return SkipFFTransformerBlock(*args, **kwargs)


class FreeNoiseTransformerBlock:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.FreeNoiseTransformerBlock instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `FreeNoiseTransformerBlock` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import FreeNoiseTransformerBlock` instead."
        )
        from .transformers.modeling_common import FreeNoiseTransformerBlock

        return FreeNoiseTransformerBlock(*args, **kwargs)


class FeedForward:
    r"""
    Backward compatibility stub. Use transformers.modeling_common.FeedForward instead.
    """

    def __new__(cls, *args, **kwargs):
        logger.warning(
            "Importing `FeedForward` from `diffusers.models.attention` is deprecated and will be removed in a future version. "
            "Please use `from diffusers.models.transformers.modeling_common import FeedForward` instead."
        )
        from .transformers.modeling_common import FeedForward

        return FeedForward(*args, **kwargs)
