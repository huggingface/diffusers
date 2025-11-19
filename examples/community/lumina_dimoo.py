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

import functools
import math
import random
import sys
from abc import abstractmethod
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union, cast
from accelerate import init_empty_weights

import numpy as np
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from PIL import Image
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from diffusers import DiffusionPipeline, VQModel
from diffusers.utils import BaseOutput, logging, replace_example_docstring
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from diffusers.pipelines.pipeline_utils import ImagePipelineOutput


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# --- Start of model definition copied from Lumina-DiMOO ---


class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"


class LayerNormType(StrEnum):
    default = "default"
    low_precision = "low_precision"
    rms = "rms"
    gemma_rms = "gemma_rms"
    amd_compatible = "amd_compatible"


class ActivationType(StrEnum):
    gelu = "gelu"
    relu = "relu"
    silu = "silu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"
    parallel = "parallel"
    llama = "llama"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    normal = "normal"
    kaiming_normal = "kaiming_normal"
    fan_in = "fan_in"
    full_megatron = "full_megatron"


@dataclass
class ModelConfig:
    """
    LLaDA (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: Optional[int] = None
    n_layers: int = 12
    mlp_ratio: int = 4
    mlp_hidden_size: Optional[int] = None
    activation_type: ActivationType = ActivationType.swiglu
    block_type: BlockType = BlockType.sequential
    block_group_size: int = 1
    alibi: bool = False
    alibi_bias_max: float = 8.0
    rope: bool = False
    rope_full_precision: bool = True
    flash_attention: bool = False
    attention_dropout: float = 0.1
    multi_query_attention: Optional[bool] = None
    attention_layer_norm: bool = False
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.1
    input_emb_norm: bool = False
    layer_norm_type: LayerNormType = LayerNormType.default
    layer_norm_with_affine: bool = True
    rms_norm_eps: float = 1e-05
    attention_layer_norm_with_affine: bool = True
    max_sequence_length: int = 1024
    rope_theta: float = 10000.0
    include_qkv_bias: Optional[bool] = False
    include_bias: bool = False
    bias_for_layer_norm: Optional[bool] = None
    scale_logits: bool = False
    vocab_size: int = 50257
    embedding_size: Optional[int] = 50304
    weight_tying: bool = True
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    mask_token_id: Optional[int] = 50256
    init_device: Optional[str] = None
    init_fn: InitFnType = InitFnType.normal
    init_std: float = 0.02
    init_cutoff_factor: Optional[float] = None
    precision: Optional[str] = None

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise Exception("You can't set `multi_query_attention` and `n_kv_heads` at the same time.")


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    two_in_three = "two_in_three"
    three_in_four = "three_in_four"
    four_in_five = "four_in_five"
    nine_in_ten = "nine_in_ten"
    fine_grained = "fine_grained"


class LLaDAConfig(PretrainedConfig):
    model_type = "llada"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.__dict__
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update({"architectures": all_kwargs.get("architectures", ["LLaDAModelLM"])})
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model


if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return functools.partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> "LayerNormBase":
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
           return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, q_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if q_mask is None:
                q_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, key_len - query_len : key_len, :],
                    pos_cos[:, :, key_len - query_len : key_len, :],
                    q_,
                )
            else:
                q_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, q_mask, :],
                    pos_cos[:, :, q_mask, :],
                    q_,
                )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> "Activation":
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class LLaDABlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

        self.use_cache = False
        self.init_cache()

    def init_cache(self):
        self.cache = {"k": {}, "v": {}, "out": {}}

    def caching(self, enable: bool = True):
        self.use_cache = enable
        self.init_cache()

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=False
            )
            return r.transpose(1, 2)
        else:
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        to_compute_mask=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        q = q.view(B, -1, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, -1, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        if self.config.rope:
            to_compute_index = (
                to_compute_mask.nonzero(as_tuple=True)[1] if self.use_cache and to_compute_mask is not None else None
            )
            q, k = self.rotary_emb(q, k, q_mask=to_compute_index)

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(attention_bias, dtype)

        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        att = att.transpose(1, 2).contiguous().view(B, T, C)

        return self.attn_out(att), None

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> "LLaDABlock":
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        x = x + self.dropout(att)

        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
           x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDALlamaBlock(LLaDABlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias, device=config.init_device
        )

        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )
        self.up_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cat="cond",
        to_compute_mask=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape

        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if use_cache:
            if cat not in self.cache["k"]:
                self.cache["k"][cat] = torch.zeros_like(x)
                self.cache["v"][cat] = torch.zeros_like(x)
            if to_compute_mask is not None:
                self.cache["k"][cat][to_compute_mask] = k.view(-1, D)
                self.cache["v"][cat][to_compute_mask] = v.view(-1, D)
                k = self.cache["k"][cat]
                v = self.cache["v"][cat]
            else:
                self.cache["k"][cat] = k
                self.cache["v"][cat] = v

        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, to_compute_mask=to_compute_mask)

        x = x + self.dropout(att)

        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = x * x_up
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDAOutput(NamedTuple):
    logits: torch.FloatTensor
    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    hidden_states: Optional[Tuple[torch.Tensor]]


class LLaDAGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    scores: torch.FloatTensor


class LLaDABlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class LLaDAModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

        self.logit_cache = {}

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        logger.info("Initializing model parameters...")
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        self.transformer.ln_f.reset_parameters()  # type: ignore

        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        use_cache=False,
        to_compute_mask=None,
        cat="",
    ) -> LLaDAOutput:
        if use_cache and to_compute_mask is not None:
            input_ids = input_ids[to_compute_mask].view(input_ids.shape[0], -1)

        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        x = self.transformer.emb_drop(x)  # type: ignore

        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None

        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        all_hidden_states = []

        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    x, _ = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, layer_past=layer_past, to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat
                    )
                else:
                    LLaDALlamaBlock.forward
                    x, _ = block(
                        x, attention_bias=attention_bias, layer_past=layer_past, to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat
                    )
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, _ = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, to_compute_mask=to_compute_mask, use_cache=use_cache, cat=cat
                )

        if last_logits_only:
            x = x[:, -1, :].unsqueeze(1)

        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            all_hidden_states.append(x)

        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        if use_cache:
            if cat not in self.logit_cache:
                self.logit_cache[cat] = torch.zeros_like(logits)
            if to_compute_mask is not None:
                self.logit_cache[cat][to_compute_mask] = logits.view(-1, logits.shape[-1])
                logits = self.logit_cache[cat]
            else:
                self.logit_cache[cat] = logits

        return LLaDAOutput(
            logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None
        )  # type: ignore[arg-type]

    def caching(self, enable: bool = True):
        LLaDABlock.caching
        for block in self.transformer.blocks:
            block.caching(enable)
        self.logit_cache = {}

    def empty_cache(self):
        for block in self.transformer.blocks:
            block.init_cache()
        self.logit_cache = {}


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(PreTrainedModel):
    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[Cache] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
        use_cache=False,
        to_compute_mask=None,
        cat="",
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            to_compute_mask=to_compute_mask,
            cat=cat,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            import warnings

            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte

    def caching(self, enable: bool = True):
        self.model.caching(enable)

    def empty_cache(self):
        self.model.empty_cache()




def create_attention_mask(original_lengths, max_tokens, device):
    batch_size = len(original_lengths)
    attention_mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=device)
    for i, length in enumerate(original_lengths):
        attention_mask[i, :length] = 1
    return attention_mask


class LLaDAForMultiModalGeneration(LLaDAModelLM):
    config_class = LLaDAConfig
    base_model_prefix = "model"

    def __init__(self, config: LLaDAConfig, *args, **kwargs):
        logger.info(f"Initializing MMadaModelLM with config: {config}")
        super().__init__(config, *args, **kwargs)

    def forward(self, input_ids=None, labels=None, infer=False, use_cache=False, to_compute_mask=None, cat="", **kwargs):
        input_ids = input_ids.tolist()
        max_tokens = max([len(_) for _ in input_ids])
        original_lengths = [len(example) for example in input_ids]
        input_ids = [example + [0] * (max_tokens - len(example)) for example in input_ids]
        input_ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        attention_mask = create_attention_mask(original_lengths, max_tokens, self.device)

        output = LLaDAModelLM.forward(
            self, input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, to_compute_mask=to_compute_mask, cat=cat
        )
        if infer:
            return output

    def get_fsdp_wrap_module_list(self) -> List:
        modules = [*list(self.model.transformer.blocks), self.model.transformer.ff_out]
        return modules


AutoConfig.register("llada", LLaDAConfig)

# --- End of model definition ---

EXAMPLE_DOC_STRING = """
Examples:
```py
>>> import torch
>>> from PIL import Image
>>> from diffusers import VQModel, DiffusionPipeline
>>> from transformers import AutoTokenizer
>>> from diffusers.utils import load_image

>>> CHECKPOINT = "Alpha-VLLM/Lumina-DiMOO"

>>> # Load VQ-VAE and tokenizer
>>> vqvae = VQModel.from_pretrained(CHECKPOINT, subfolder="vqvae").to(device=device, dtype=torch_dtype)
>>> tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)

>>> # Initialize the Lumina-DiMOO pipeline
>>> pipe = DiffusionPipeline.from_pretrained(
...     CHECKPOINT,
...     custom_pipeline="lumina_dimoo",
...     vqvae=vqvae,
...     tokenizer=tokenizer,
...     torch_dtype=torch.bfloat16
... )
>>> pipe.to("cuda")

>>> # Load input image
>>> input_image  = Image.open("path/to/your/ref_image.png").convert("RGB")

>>> prompt = (
...     " your prompt. "
... )

>>> # Run image-to-image generation
>>> out = pipe(
...     prompt=prompt,
...     image=input_image,
...     edit_type="depth_control",
...     num_inference_steps=64,
...     task="image_to_image",
... )

>>> out.images[0].save("i2i_test_output.png")
"""


# --- Helper functions ---


def cosine_schedule(t):
    return torch.cos(t * math.pi / 2)


def gumbel_noise(t: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if generator is None:
        u = torch.rand_like(t)
    else:
        u = torch.rand(t.shape, device=t.device, dtype=t.dtype, generator=generator)
    return -torch.log(-torch.log(u + 1e-20) + 1e-20)


def add_gumbel_noise(logits, temperature):
    """
    Gumbel noise addition function
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality
    Therefore using float64
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def gumbel_max_sample(logits, temperature=1.0, generator=None):
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    gumbel_noise_ = gumbel_noise(logits, generator=generator)
    return torch.argmax(logits / temperature + gumbel_noise_, dim=-1)

def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals
    Since LLaDA employs a linear noise schedule (as defined in Eq.(8)),
    the expected number of tokens transitioned at each step should be consistent
    
    This function is designed to precompute the number of tokens that need to be transitioned at each step
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def mask_by_random_topk(keep_n, probs, temperature=1.0, generator=None):
    B, S = probs.shape
    noise = gumbel_noise(probs, generator=generator)

    conf = probs / temperature + noise

    mask = torch.zeros_like(conf, dtype=torch.bool)
    for i in range(B):
        k = keep_n[i]
        if k > 0:
            top_k_indices = torch.topk(conf[i], k, largest=True).indices
            mask[i, top_k_indices] = True
    return mask


def calculate_vq_params(height, width, vae_scale_factor=32):
    token_grid_height = height // vae_scale_factor
    token_grid_width = width // vae_scale_factor
    seq_len = token_grid_height * token_grid_width
    newline_every = token_grid_width
    return seq_len, newline_every, token_grid_height, token_grid_width


def add_break_line(tokens, token_grid_height, token_grid_width, new_number):
    new_tokens = []
    for i in range(token_grid_height):
        start = i * token_grid_width
        end = (i + 1) * token_grid_width
        row = tokens[start:end]
        new_tokens.extend(row)
        if i < token_grid_height - 1:
            new_tokens.append(new_number)
    return new_tokens


def generate_crop_size_list(num_patches, patch_size, max_ratio=4.0):
    assert max_ratio >= 1.0
    crop_size_list = []
    wp, hp = num_patches, 1
    while wp > 0:
        if max(wp, hp) / min(wp, hp) <= max_ratio:
            crop_size_list.append((wp * patch_size, hp * patch_size))
        if (hp + 1) * wp <= num_patches:
            hp += 1
        else:
            wp -= 1
    return crop_size_list


def center_crop(pil_image, crop_size):
    while pil_image.size[0] >= 2 * crop_size[0] and pil_image.size[1] >= 2 * crop_size[1]:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = max(crop_size[0] / pil_image.size[0], crop_size[1] / pil_image.size[1])
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    crop_left = random.randint(0, pil_image.size[0] - crop_size[0])
    crop_upper = random.randint(0, pil_image.size[1] - crop_size[1])
    crop_right = crop_left + crop_size[0]
    crop_lower = crop_upper + crop_size[1]
    return pil_image.crop(box=(crop_left, crop_upper, crop_right, crop_lower))


def var_center_crop(pil_image, crop_size_list, random_top_k=1):
    w, h = pil_image.size
    rem_percent = [min(cw / w, ch / h) / max(cw / w, ch / h) for cw, ch in crop_size_list]
    crop_size = random.choice(
        sorted(((x, y) for x, y in zip(rem_percent, crop_size_list)), reverse=True)[:random_top_k]
    )[1]
    return center_crop(pil_image, crop_size)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    w, h = image.size
    w, h = (x - x % 32 for x in (w, h))
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def encode_img_with_breaks(image, vqvae, special_tokens, vae_scale_factor: int = 16):
    """
    Encode image, add VQ offset, add newlines, and wrap with BOI/EOI tokens.
    This function mirrors the logic from the original inference script.
    """
    orig = image.convert("RGB")
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    pixels = image_processor.preprocess(orig).to(vqvae.device, dtype=vqvae.dtype)
    latents = vqvae.encode(pixels).latents

    latents_bsz, _, lat_h, lat_w = latents.shape

    quantized = vqvae.quantize(latents)[2][2] + special_tokens["image_token_offset"]
    quantized_with_offset = quantized.reshape(latents_bsz, lat_h, lat_w).flatten().tolist()

    tokens_with_breaks = add_break_line(
        quantized_with_offset, lat_h, lat_w, special_tokens["newline_token"]
    )
    return [special_tokens["boi"]] + tokens_with_breaks + [special_tokens["eoi"]]


def create_prompt_templates():
    """Create prompt templates for various tasks based on prompt_utils.py"""
    templates = {
        "text_understanding": "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer.",
        "image_generation": "Generate an image according to the text prompt.",
        "image_editing": "Generate an image applying the following editing instruction based on the original image.",
        "dense_prediction": "Perform dense prediction on the given images.",
        "control_generation": "Generate an image according to the text prompt and the given control image.",
        "subject_generation": "Generate an image according to the text prompt and the given object image.",
        "multi_view": "Generate a view-image based on the given image.",
        "style_transfer": "Transform the current image into the style of the provided image.",
    }
    return templates


def generate_image_to_image_prompt(prompt_text, edit_type, templates):
    """
    Generate prompt for image-to-image generation based on prompt_utils.py
    """
    if "dense" in edit_type or "canny_pred" in edit_type:
        des = {
            "canny": "canny edge map",
            "hed": "hed edge map",
            "normal": "normal map",
            "sam2mask": "sam2 mask",
            "depth": "depth map",
            "openpose": "pose estimation map",
        }
        system_prompt = templates["dense_prediction"]
        prompt_text_used = f"Generate a {des.get(edit_type.split('_')[0], 'dense map')} according to the image."

    elif "control" in edit_type:
        system_prompt = templates["control_generation"]
        prompt_text_used = prompt_text

    elif "subject" in edit_type:
        system_prompt = templates["subject_generation"]
        prompt_text_used = prompt_text

    elif "edit" in edit_type:
        system_prompt = templates["image_editing"]
        prompt_text_used = prompt_text

    elif "ref_transfer" in edit_type or "image_ref_transfer" in edit_type:
        system_prompt = templates["style_transfer"]
        prompt_text_used = "Transform the current image into the style of the provided image."

    elif "multi_view" in edit_type:
        system_prompt = templates["multi_view"]
        prompt_text_used = f"Generate the {edit_type.split('_')[-1]} view based on the provided front view."

    else:
        system_prompt = "Generate an image according to the prompt and image."
        prompt_text_used = prompt_text

    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + prompt_text_used + "</user>"
    uncon_prompt = "<system>" + system_prompt + "</system>" + "<user>" + "<uncondition>" + "</user>"

    return input_prompt, uncon_prompt

def generate_text_to_image_prompt(prompt_text: str, templates: Optional[Dict] = None) -> Tuple[str, str]:
    """
    Generate prompt for text-to-image generation
    
    Args:
        prompt_text: User input text prompt
        templates: Optional prompt templates dict
        
    Returns:
        Tuple of (input_prompt, unconditional_prompt)
    """
    if templates is None:
        templates = create_prompt_templates()
    
    system_prompt = templates["image_generation"]
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + prompt_text + "</user>"
    uncon_prompt = "<system>" + system_prompt + "</system>" + "<user>" + "<uncondition>" + "</user>"
    
    return input_prompt, uncon_prompt


def generate_multimodal_understanding_prompt(question: str, templates: Optional[Dict] = None) -> str:
    """
    Generate prompt for multimodal understanding (MMU)
    
    Args:
        question: User question about the image
        templates: Optional prompt templates dict
        
    Returns:
        Formatted input prompt
    """
    if templates is None:
        templates = create_prompt_templates()
    
    system_prompt = "You are a multimodal model that can process both text and images. Answer the following question based on the provided images. Analyze each image and combine relevant details to answer."
    input_prompt = "<system>" + system_prompt + "</system>" + "<user>" + question + "</user>"
    
    return input_prompt


@torch.no_grad()
def encode_img_with_paint(
    img: Image.Image,
    vqvae: VQModel,
    *,
    mask_h_ratio: float = 1,   # Height ratio
    mask_w_ratio: float = 0.2,    # Width ratio
    gray_value: int = 127,        # Visualization gray value
    downsample_mode: str = "area",# Pixel mask alignment to latent grid
    dilate_latent_k: int = 0,     # Optional dilation on latent grid (grid count)
    mask_mode: str = "inpainting",   # "inpainting" | "outpainting"
    special_tokens
):
    """
    Encode image with mask for inpainting/outpainting tasks
    
    Args:
        img: Input PIL image
        vqvae: VQ-VAE model for encoding
        mask_h_ratio: Height ratio for mask region (default: 1.0)
        mask_w_ratio: Width ratio for mask region (default: 0.2)
        gray_value: Gray value for mask visualization (default: 127)
        downsample_mode: Downsampling mode for mask alignment ("area", "nearest", "bilinear")
        dilate_latent_k: Dilation kernel size for latent grid (default: 0)
        mask_mode: Mask mode - "inpainting" (mask inside) or "outpainting" (mask outside)
    
    Returns:
        img_token: List[int] - Token sequence with newlines (126084) inserted at row ends;
                              masked positions = 126336, others = index + 126356
        vis_img: PIL.Image - Gray mask visualization image (consistent with mask_mode)
    
    Note:
        * Encoding uses original image strictly; mask only maps to latent grid to determine
          which tokens are set to MASK_TOKEN_ID.
        * mask_mode="inpainting": mask inside rectangle; "outpainting": mask outside rectangle (inverse).
    """

    assert mask_mode in ("inpainting", "outpainting"), "mask_mode must be 'inpainting' or 'outpainting'"

    # --- 1) Calculate center rectangle and generate visualization ---
    img = img.convert("RGB")
    W, H = img.size
    mh = int(round(H * mask_h_ratio))
    mw = int(round(W * mask_w_ratio))
    top = (H - mh) // 2
    left = (W - mw) // 2
    bottom = top + mh
    right = left + mw

    if mask_mode == "inpainting":
        vis_img = img.copy()
        draw = ImageDraw.Draw(vis_img)
        draw.rectangle([left, top, right, bottom], fill=(gray_value, gray_value, gray_value))
    elif mask_mode == "outpainting":  # outpainting
        bg = Image.new("RGB", (W, H), (gray_value, gray_value, gray_value))
        crop = img.crop((left, top, right, bottom))
        bg.paste(crop, (left, top))
        vis_img = bg

    # --- 2) VQ encoding using original image ---
    vae_scale_factor = 2 ** (len(vqvae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor, do_normalize=False)
    x = image_processor.preprocess(img).to(vqvae.device)  # 1 x 3 x H' x W'
    latents = vqvae.encode(x).latents                     # 1 x C x h x w
    _, _, lat_h, lat_w = latents.shape

    # Quantization indices
    quant_pack = vqvae.quantize(latents)
    indices = quant_pack[2][2].view(1, lat_h, lat_w)      # 1 x h x w, long

    # --- 3) Pixel mask -> latent grid mask (aligned with encoding input size) ---
    Hp, Wp = x.shape[-2:]
    mask_px = torch.zeros((1, 1, Hp, Wp), dtype=torch.float32, device=vqvae.device)
    # First generate mask where "rectangle inside=1, outside=0"
    top_p  = int(round(top  * Hp / H))
    left_p = int(round(left * Wp / W))
    bh_p   = int(round(mh   * Hp / H))
    bw_p   = int(round(mw   * Wp / W))
    mask_px[:, :, top_p:top_p+bh_p, left_p:left_p+bw_p] = 1.0

    # If outpainting, need to invert (outside=1, inside=0 is the masked region)
    if mask_mode == "outpainting":
        mask_px = 1.0 - mask_px

    if downsample_mode not in ("nearest", "area", "bilinear"):
        downsample_mode = "area"
    mask_lat = F.interpolate(mask_px, size=(lat_h, lat_w), mode=downsample_mode)
    mask_lat = (mask_lat > 0.5) if downsample_mode == "area" else (mask_lat >= 0.5)
    mask_lat = mask_lat[0, 0]        # h x w (bool)

    # Optional: latent grid dilation (after inversion is applied)
    if dilate_latent_k > 0:
        m = mask_lat.float().unsqueeze(0).unsqueeze(0)
        ker = 2 * dilate_latent_k + 1
        m = F.max_pool2d(m, kernel_size=ker, stride=1, padding=dilate_latent_k)
        mask_lat = (m[0, 0] > 0.5)

    # --- 4) Generate tokens: masked positions=MASK_TOKEN_ID, others=indices+VQ_OFFSET ---
    idx_flat = indices.view(-1)
    mask_flat = mask_lat.view(-1)
    tokens = torch.empty_like(idx_flat)
    tokens[mask_flat] = special_tokens['mask_token']
    tokens[~mask_flat] = idx_flat[~mask_flat] + special_tokens['image_token_offset']
    tokens_list = tokens.tolist()

    # --- 5) Insert newlines (no longer wrapped in <boi>/<eoi>, consistent with current return) ---

    img_token = add_break_line(tokens_list, lat_h, lat_w, special_tokens['newline_token'])
    return img_token, vis_img


class LuminaDiMOOPipelineOutput(BaseOutput):
    """
    Output class for the Lumina-DiMOO pipeline.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`, *optional*):
            List of generated PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        text (`str`, *optional*):
            Generated text from the multimodal understanding task.
    """

    images: Optional[Union[List[Image.Image], np.ndarray]] = None
    text: Optional[str] = None


class LuminaDiMOOPipeline(DiffusionPipeline):
    """
    A unified pipeline for Text-to-Image, Image-to-Image, and Multimodal Understanding
    using the Lumina-DiMOO model.

    This model was contributed by https://huggingface.co/Alpha-VLLM

    Args:
        llm ([`LLaDAForMultiModalGeneration`]):
            The core LLM for multimodal generation, e.g., `LLaDAForMultiModalGeneration`.
        vqvae ([`VQModel`]):
            Vector Quantized Variational Auto-Encoder (VQ-VAE) model to encode and decode images to and from discrete
            latent representations.
        tokenizer ([`AutoTokenizer`):
            An `AutoTokenizer` to tokenize text prompts.
    """

    def __init__(
        self,
        vqvae: VQModel,
        tokenizer: AutoTokenizer,
        checkpoint: Optional[str] = "Alpha-VLLM/Lumina-DiMOO",
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: Optional[str] = "auto",         
        low_cpu_mem_usage: bool = True, 
    ):
        super().__init__()
        self.register_modules(
            vqvae=vqvae,
            tokenizer=tokenizer,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vqvae.config.block_out_channels) - 1)
        self.special_tokens = {
            "mask_token": 126336,
            "newline_token": 126084,
            "boa": 126354,
            "eoa": 126355,
            "boi": 126349,
            "eoi": 126350,
            "image_token_offset": 126356,
            "uncondition":126351
        }
        self.prompt_templates = create_prompt_templates()

        # If checkpoint is not provided, reuse the model path from from_pretrained
        if checkpoint is None:
            checkpoint = self._name_or_path
            raise ValueError("A `checkpoint` path must be provided to load the LLM, either directly or via `from_pretrained`.")

        print("[Lumina] start loading LLaDA ...")
        self.llm = LLaDAForMultiModalGeneration.from_pretrained(
            checkpoint, torch_dtype=torch_dtype, trust_remote_code=True,
            device_map=device_map,                   
            low_cpu_mem_usage=low_cpu_mem_usage,    
            use_safetensors=True,
        )
        print("   LlaDA Loaded Successfully.")

    @staticmethod
    @torch.no_grad()
    def generate_i2i(
        model: LLaDAForMultiModalGeneration,
        prompt: torch.LongTensor,
        *,
        seq_len: int = 1024,
        newline_every: int = 16,
        timesteps: int = 18,
        mask_token_id: int = 126336,
        newline_id: int = 126084,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        cfg_img: float = 0.0,
        uncon_text: torch.LongTensor,
        uncon_image: torch.LongTensor,
        code_start: Optional[int] = None,
        codebook_size: int = 8192,
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
        text_vocab_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.LongTensor:
        """
        Image-to-image MaskGit generation (supports CFG for text and image)
        
        Args:
            model: Model
            prompt: Prompt tensor
            seq_len: Sequence length
            newline_every: Newline interval per row
            timesteps: Number of timesteps
            mask_token_id: Mask token id
            newline_id: Newline token id
            temperature: Temperature
            cfg_scale: Text CFG scale
            cfg_img: Image CFG scale
            code_start: Prediction image token satrt index
            uncon_text: Unconditional text input
            uncon_image: Unconditional image input
            codebook_size: Codebook size
            noise_schedule: Noise schedule function
            text_vocab_size: Text vocabulary size
            generator: Random number generator
        
        Returns:
            Final VQ codes (1, seq_len)
        """
        device = next(model.parameters()).device
        prompt = prompt.to(device)
        B, P = prompt.shape
        assert B == 1, "batch>1 not supported – wrap in loop if needed"
        
        x = prompt

        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
        vq_len = unknown_cnt

        # Infer text vocabulary size
        if text_vocab_size is None:
            vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
            text_vocab_size = vocab_total - codebook_size
        vocab_offset = text_vocab_size

        for step in range(timesteps):
            if unknown_cnt.item() == 0:
                break

            # Calculate number of tokens to keep (continue masking) this round
            if step < timesteps - 1:
                frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
                keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
            else:
                keep_n = torch.zeros_like(unknown_cnt)

            # Forward pass (with/without CFG)
            if cfg_scale > 0 or cfg_img > 0:
                # CFG text
                uncond_text = torch.cat((uncon_text.to(x.device), x[:, code_start-2:]), dim=1)
                uncond_text_vq_mask = torch.cat((torch.zeros((1, uncon_text.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start-2:]), dim=1)
                # CFG image
                uncond_img = torch.cat((uncon_image.to(x.device), x[:, code_start-2:]), dim=1)
                uncond_img_vq_mask = torch.cat((torch.zeros((1, uncon_image.size(1)), dtype=torch.bool, device=x.device), vq_mask[:, code_start-2:]), dim=1)

                cond_logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]
                uncond_logits_text = model(uncond_text, infer=True).logits[:, uncond_text_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
                uncond_logits_img = model(uncond_img, infer=True).logits[:, uncond_img_vq_mask[0], vocab_offset : vocab_offset + codebook_size]
                logits = cond_logits + cfg_scale * (cond_logits - uncond_logits_text) + cfg_img * (cond_logits - uncond_logits_img)
            else:
                logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

            sampled = gumbel_max_sample(logits, temperature, generator=generator)
            sampled_full = sampled + vocab_offset
            probs = torch.softmax(logits, dim=-1)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
            x.view(-1)[flat_idx] = sampled_full.view(-1)

            conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
            conf_map.view(-1)[flat_idx] = conf.view(-1)

            mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
            x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
            vq_mask = x == mask_token_id
            unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

        # Remove newline tokens
        vq_ids = x[0, code_start:-2]
        vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
        return vq_ids


    @staticmethod
    @torch.no_grad()
    def generate_image(
        model: LLaDAForMultiModalGeneration,
        prompt: torch.LongTensor,
        *,
        seq_len: int = 1024,
        newline_every: int = 16,
        timesteps: int = 18,
        mask_token_id: int = 126336,
        newline_id: int = 126084,
        temperature: float = 1.0,
        cfg_scale: float = 0.0,
        uncon_ids: torch.LongTensor,
        code_start: Optional[int] = None,
        codebook_size: int = 8192,
        noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
        text_vocab_size: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        use_cache=True,
        cache_ratio=0.9,
        refresh_interval=5,
        warmup_ratio=0.3
    ) -> torch.LongTensor:
        """
        MaskGit parallel decoding to generate VQ tokens
        
        Args:
            model: Model
            prompt: Prompt tensor
            seq_len: Sequence length
            newline_every: Newline interval per row
            timesteps: Number of timesteps
            mask_token_id: Mask token id
            newline_id: Newline token id
            temperature: Temperature
            cfg_scale: CFG scale
            uncon_ids: Unconditional input
            code_start: Image token satrt index
            codebook_size: Codebook size
            noise_schedule: Noise schedule function
            text_vocab_size: Text vocabulary size
            generator: Random number generator
        
        Returns:
            Final VQ codes (1, seq_len)
        """


        device = next(model.parameters()).device
        prompt = prompt.to(device)
        B, P = prompt.shape
        assert B == 1, "batch>1 not supported – wrap in loop if needed"

        x = prompt
        
        vq_mask = x == mask_token_id
        unknown_cnt = vq_mask.sum(dim=1, keepdim=True)
        vq_len = unknown_cnt

        if isinstance(model, LLaDAForMultiModalGeneration):
            model.caching(use_cache)
        else:  # DDP
            model.module.caching(use_cache)

        warmup_step = int(timesteps * warmup_ratio)
        refresh_steps = torch.zeros(timesteps, dtype=torch.bool)
        for step in range(timesteps):
            if not use_cache or step <= warmup_step or (step-warmup_step) % refresh_interval == 0:
                refresh_steps[step] = True
        compute_ratio = 1 - cache_ratio

        # Infer text vocabulary size
        if text_vocab_size is None:
            vocab_total = model(torch.zeros(1, 1, dtype=torch.long, device=device), infer=True).logits.size(-1)
            text_vocab_size = vocab_total - codebook_size
        vocab_offset = text_vocab_size

        for step in range(timesteps):
            if unknown_cnt.item() == 0:
                break

            # Calculate number of tokens to keep (continue masking) this round
            if step < timesteps - 1:
                frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
                keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
            else:
                keep_n = torch.zeros_like(unknown_cnt)

            if use_cache and step and refresh_steps[step]:
                if isinstance(model, LLaDAForMultiModalGeneration):
                    model.empty_cache()
                else:  # DDP
                    model.module.empty_cache()

            # Forward pass (with/without CFG)
            if cfg_scale > 0:
                import time
                t0 = time.time()
                uncond = torch.cat((uncon_ids.to(x.device), x[:, code_start-2:]), axis=1)
                uncond_vq_mask = torch.cat((torch.zeros((1, uncon_ids.size()[1]), dtype=torch.bool).to(x.device), vq_mask[:, code_start-2:]), axis=1)
                cond_logits = model(x, infer=True,
                        cat='cond', use_cache=use_cache, 
                        to_compute_mask = cond_to_compute_mask if not refresh_steps[step] else None,
                    ).logits[..., vocab_offset : vocab_offset + codebook_size]
                cond_mask_logits = cond_logits[vq_mask].view(B, -1, codebook_size)
                uncond_logits = model(uncond, infer=True,
                        cat='uncond', use_cache=use_cache, 
                        to_compute_mask = uncond_to_compute_mask if not refresh_steps[step] else None
                    ).logits[..., vocab_offset : vocab_offset + codebook_size]
                uncond_mask_logits = uncond_logits[uncond_vq_mask].view(B, -1, codebook_size)
                logits = (1 + cfg_scale) * cond_mask_logits - cfg_scale * uncond_mask_logits
            else:
                logits = model(x, infer=True).logits[:, vq_mask[0], vocab_offset : vocab_offset + codebook_size]

            sampled = gumbel_max_sample(logits, temperature, generator=generator)
            sampled_full = sampled + vocab_offset
            probs = torch.softmax(logits, dim=-1)
            conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            flat_idx = vq_mask.nonzero(as_tuple=False)[:, 1]
            x.view(-1)[flat_idx] = sampled_full.view(-1)

            conf_map = torch.full_like(x, -math.inf, dtype=probs.dtype)
            conf_map.view(-1)[flat_idx] = conf.view(-1)

            mask_sel = mask_by_random_topk(keep_n.squeeze(1), conf, temperature=temperature, generator=generator)
            x.view(-1)[flat_idx[mask_sel.view(-1)]] = mask_token_id
            vq_mask = x == mask_token_id
            unknown_cnt = vq_mask.sum(dim=1, keepdim=True)

            if use_cache and step < timesteps - 1 and not refresh_steps[step+1]:
                cond_conf = cond_logits.max(dim=-1)[0]
                cond_conf_threshold = torch.quantile(cond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
                cond_to_compute_mask = cond_conf <= cond_conf_threshold

                uncond_conf = uncond_logits.max(dim=-1)[0]
                uncond_conf_threshold = torch.quantile(uncond_conf.to(torch.float), compute_ratio, dim=-1, keepdim=True)
                uncond_to_compute_mask = uncond_conf <= uncond_conf_threshold
                
        # Remove newline tokens
        vq_ids = x[0, code_start:-2]
        vq_ids = vq_ids[vq_ids != newline_id].view(1, seq_len)
        return vq_ids


    @staticmethod
    @torch.no_grad()
    def generate_text_understanding(
        model: LLaDAForMultiModalGeneration,
        prompt, 
        steps=128, 
        gen_length=128, 
        block_length=128, 
        temperature=0.,
        cfg_scale=0., 
        remasking='low_confidence', 
        mask_id=126336, 
        code_start: Optional[int] = None,
    ):
        """
        Text understanding generation function
        
        Args:
            model: Mask predictor
            prompt: Input prompt tensor (1, L)
            steps: Sampling steps, less than or equal to gen_length
            gen_length: Generated answer length
            block_length: Block length, less than or equal to gen_length
            temperature: Categorical distribution sampling temperature
            cfg_scale: Unsupervised classifier-free guidance scale
            remasking: Remasking strategy 'low_confidence' or 'random'
            mask_id: The token id of [MASK] is 126336
            code_start: Prediction text token satrt index
        """
        device = next(model.parameters()).device

        x = prompt

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (x[:, code_start + num_block * block_length: code_start + (num_block + 1) * block_length:] == mask_id)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
            
            for i in range(steps):
                mask_index = (x == mask_id)
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_, infer=True).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, infer=True).logits

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

                if remasking == 'low_confidence':
                    p = F.softmax(logits.to(torch.float64), dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, code_start + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
            
        
        return x



    @torch.no_grad()
    def _image_to_image(
        self,
        prompt: str,
        image: Union[Image.Image, str],
        ref_image: Optional[PipelineImageInput] = None,
        edit_type: str = "canny_pred",
        num_inference_steps: int = 64,
        temperature: float = 1.0,
        cfg_scale: float = 2.5,
        cfg_img: float = 4.0,
        output_type: Optional[str] = "pil",
    ):
        
        if isinstance(prompt, list):
            raise ValueError("Batching is not supported for this pipeline.")

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        if isinstance(ref_image, str):
            ref_image = Image.open(ref_image).convert("RGB")

        input_prompt, uncon_text = generate_image_to_image_prompt(prompt, edit_type, self.prompt_templates)

        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        
        # Correctly encode input images with newline tokens
        if "image_ref_transfer" in edit_type:
            if ref_image is None:
                raise ValueError("`ref_image` must be provided for `image_ref_transfer` edit type.")
            processed_img = var_center_crop(image, crop_size_list=crop_size_list)
            input_img_token = encode_img_with_breaks(processed_img, self.vqvae, self.special_tokens)

            referring_img = var_center_crop(ref_image, crop_size_list=crop_size_list)
            referring_img_token = encode_img_with_breaks(referring_img, self.vqvae, self.special_tokens)

            image_width, image_height = referring_img.size
            seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
                referring_img.height, referring_img.width, self.vae_scale_factor
            )
        else:
            processed_img = var_center_crop(image, crop_size_list=crop_size_list)
            input_img_token = encode_img_with_breaks(processed_img, self.vqvae, self.special_tokens)
            image_width, image_height = processed_img.size
            seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
                processed_img.height, processed_img.width, self.vae_scale_factor
            )

        prompt_ids = self.tokenizer(input_prompt)["input_ids"]
        uncon_text_ids = self.tokenizer(uncon_text)["input_ids"]

        img_mask_token = add_break_line(
            [self.special_tokens["mask_token"]] * seq_len,
            token_grid_height,
            token_grid_width,
            new_number=self.special_tokens["newline_token"],
        )
        img_pred_token = (
            [self.special_tokens["boa"]]
            + [self.special_tokens["boi"]]
            + img_mask_token
            + [self.special_tokens["eoi"]]
            + [self.special_tokens["eoa"]]
        )

        if "image_ref_transfer" in edit_type:
            con_input = prompt_ids[:-1] + input_img_token + referring_img_token + prompt_ids[-1:]
            uncon_input_text = uncon_text_ids[:-1] + input_img_token + referring_img_token + uncon_text_ids[-1:]
        else:
            con_input = prompt_ids[:-1] + input_img_token + prompt_ids[-1:]
            uncon_input_text = uncon_text_ids[:-1] + input_img_token + uncon_text_ids[-1:]
        uncon_input_image = prompt_ids

        code_start = len(con_input) + 2

        con_input = torch.tensor(con_input + img_pred_token, device=self.device).unsqueeze(0)
        uncon_input_text = torch.tensor(uncon_input_text, device=self.device).unsqueeze(0)
        uncon_input_image = torch.tensor(uncon_input_image, device=self.device).unsqueeze(0)

        vq_tokens = self.generate_i2i(
            self.llm,
            con_input,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=num_inference_steps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            cfg_img=cfg_img,
            uncon_text=uncon_input_text,
            uncon_image=uncon_input_image,
            code_start=code_start
        )

        if vq_tokens.shape[1] != token_grid_height * token_grid_width:
            raise ValueError(
                f"VQ codes length mismatch: {vq_tokens.shape[1]} != {token_grid_height * token_grid_width} "
                f"for image size ({image_height},{image_width}) with scale {self.vae_scale_factor}"
            )

        latents = (
            vq_tokens.view(1, token_grid_height, token_grid_width).to(self.vqvae.device) - self.special_tokens["image_token_offset"]
        ).long()

        shape = (1, token_grid_height, token_grid_width, self.vqvae.config.latent_channels)

        recon = self.vqvae.decode(
            latents,
            force_not_quantize=True,
            shape=shape,
        ).sample.clip(0, 1)

        img_proc = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)
        image = img_proc.postprocess(recon.detach(), output_type=output_type)

        return image

    @torch.no_grad()
    def _text_to_image(
        self,
        prompt: str,
        height: int,
        width: int,
        painting_mode: Optional[str] = None,
        painting_image: Optional[PipelineImageInput] = None,
        cfg_scale: float = 4.0,
        use_cache: bool = True,
        cache_ratio: float = 0.9,
        refresh_interval: int = 5,
        warmup_ratio: float = 0.3,
        num_inference_steps: int = 64,
        temperature: float = 1.0,
        mask_h_ratio: float = 1.0,
        mask_w_ratio: float = 0.2
    ):
        if isinstance(painting_image, str):
            painting_image = Image.open(painting_image)

        if painting_mode and painting_image:
            width, height = painting_image.size

        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(height, width, self.vae_scale_factor)

        input_prompt, uncon_prompt = generate_text_to_image_prompt(prompt, self.prompt_templates)

        con_prompt_token = self.tokenizer(input_prompt)["input_ids"]
        uncon_prompt_token = self.tokenizer(uncon_prompt)["input_ids"]

        if painting_mode:
            img_mask_token, img_vis = encode_img_with_paint(
                painting_image,
                vqvae=self.vqvae,
                mask_h_ratio=mask_h_ratio,
                mask_w_ratio=mask_w_ratio,
                mask_mode=painting_mode,
                special_tokens=self.special_tokens,
            )
        else:
            img_mask_token = add_break_line(
                [self.special_tokens["mask_token"]] * seq_len,
                token_grid_height,
                token_grid_width,
                new_number=self.special_tokens["newline_token"],
            )

        img_pred_token = (
            [self.special_tokens["boa"]]
            + [self.special_tokens["boi"]]
            + img_mask_token
            + [self.special_tokens["eoi"]]
            + [self.special_tokens["eoa"]]
        )

        prompt_ids = torch.tensor(con_prompt_token + img_pred_token, device=self.device).unsqueeze(0)
        uncon_ids = torch.tensor(uncon_prompt_token, device=self.device).unsqueeze(0)

        code_start = len(con_prompt_token) + 2

        vq_tokens = self.generate_image(
            model=self.llm,
            prompt=prompt_ids,
            seq_len=seq_len,
            newline_every=newline_every,
            timesteps=num_inference_steps,
            temperature=temperature,
            cfg_scale=cfg_scale,
            uncon_ids=uncon_ids,
            code_start=code_start,
            use_cache=use_cache,
            cache_ratio=cache_ratio,
            refresh_interval=refresh_interval,
            warmup_ratio=warmup_ratio
        )

        latents = (
            vq_tokens.view(1, token_grid_height, token_grid_width).to(self.vqvae.device) - self.special_tokens["image_token_offset"]
        ).long()

        shape = (1, token_grid_height, token_grid_width, self.vqvae.config.latent_channels)
        recon = self.vqvae.decode(latents, force_not_quantize=True, shape=shape).sample.clip(0, 1)

        img_proc = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_normalize=False)
        image = img_proc.postprocess(recon.detach(), output_type="pil")

        return image

    @torch.no_grad()
    def _multimodal_understanding(
        self,
        prompt: str,
        image: PipelineImageInput,
        num_inference_steps: int = 128,
        gen_length: int = 1024,
        block_length: int = 128,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
    ):

        if isinstance(image, str):
            image = Image.open(image)

        input_prompt = generate_multimodal_understanding_prompt(prompt)
        input_ids = self.tokenizer(input_prompt)["input_ids"]

        crop_size_list = generate_crop_size_list((1024 // 32) ** 2, 32)
        processed_image = var_center_crop(image, crop_size_list=crop_size_list)
        
        image_width, image_height = processed_image.size
        seq_len, newline_every, token_grid_height, token_grid_width = calculate_vq_params(
            image_height, image_width, self.vae_scale_factor
        )

        input_img_token = encode_img_with_breaks(processed_image, self.vqvae, self.special_tokens)

        input_token = input_ids[:-1] + input_img_token + input_ids[-1:]
        code_start = len(input_token) + 1

        input_token = input_token + [self.special_tokens["boa"]] + gen_length * [self.special_tokens["mask_token"]] + [self.special_tokens["eoa"]]
        input_ids = torch.tensor(input_token, device=self.device).unsqueeze(0)

        output_tokens = self.generate_text_understanding(
            model=self.llm,
            prompt=input_ids,
            steps=num_inference_steps,
            gen_length=gen_length,
            block_length=block_length, 
            cfg_scale=cfg_scale, 
            temperature=temperature,
            remasking=remasking,
            code_start=code_start
        )

        generated_text = self.tokenizer.batch_decode(output_tokens[:, code_start:-1], skip_special_tokens=True)[0]
        return generated_text

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str,
        image: Optional[PipelineImageInput] = None,
        task: str = "auto",
        **kwargs,
    ) -> LuminaDiMOOPipelineOutput:
        r"""
        Unified entry for 'text_to_image' | 'image_to_image' | 'multimodal_understanding'.

        Examples:
        {EXAMPLE_DOC_STRING}
        """
        if task == "auto":
            if image is None:
                task = "text_to_image"
            elif "edit_type" in kwargs:
                task = "image_to_image"
            else:
                task = "multimodal_understanding"

        if task == "text_to_image":
            # Default values from inference_t2i.py
            t2i_kwargs = {
                "height": kwargs.pop("height", 1024),
                "width": kwargs.pop("width", 1024),
                "num_inference_steps": kwargs.pop("num_inference_steps", 64), 
                "cfg_scale": kwargs.pop("cfg_scale", 4.0), 
                "temperature": kwargs.pop("temperature", 1.0),
                "painting_mode": kwargs.pop("painting_mode", None),
                "painting_image": kwargs.pop("painting_image", None),
                "mask_h_ratio": kwargs.pop("mask_h_ratio", 1.0),
                "mask_w_ratio": kwargs.pop("mask_w_ratio", 0.2),
                "use_cache": kwargs.pop("use_cache", True),
                "cache_ratio": kwargs.pop("cache_ratio", 0.9),
                "refresh_interval": kwargs.pop("refresh_interval", 5),
                "warmup_ratio": kwargs.pop("warmup_ratio", 0.3),
            }
            images = self._text_to_image(prompt=prompt, **t2i_kwargs)
            return LuminaDiMOOPipelineOutput(images=images, text=None)

        elif task == "image_to_image":
            if image is None:
                raise ValueError("`image` must be provided for image_to_image task.")
            i2i_kwargs = {
                "ref_image": kwargs.pop("ref_image", None),
                "edit_type": kwargs.pop("edit_type", "canny_pred"),
                "num_inference_steps": kwargs.pop("num_inference_steps", 64), 
                "temperature": kwargs.pop("temperature", 1.0),
                "cfg_scale": kwargs.pop("cfg_scale", 2.5), 
                "cfg_img": kwargs.pop("cfg_img", 4.0),
            }
            images = self._image_to_image(prompt=prompt, image=image, **i2i_kwargs)
            return LuminaDiMOOPipelineOutput(images=images, text=None)

        elif task == "multimodal_understanding":
            if image is None:
                raise ValueError("`image` must be provided for multimodal_understanding task.")
            mmu_kwargs = {
                "num_inference_steps": kwargs.pop("num_inference_steps", 128), 
                "gen_length": kwargs.pop("gen_length", 1024),
                "block_length": kwargs.pop("block_length", 256),
                "temperature": kwargs.pop("temperature", 0.0),
                "cfg_scale": kwargs.pop("cfg_scale", 0.0),
                "remasking": kwargs.pop("remasking", "low_confidence"),
            }
            text = self._multimodal_understanding(prompt=prompt, image=image, **mmu_kwargs)
            return LuminaDiMOOPipelineOutput(images=None, text=text)

        else:
            raise ValueError(f"Unknown task: {task}. Supported tasks are 'text_to_image', 'image_to_image', 'multimodal_understanding', and 'auto'.")
