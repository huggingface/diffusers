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

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..models.attention import AttentionModuleMixin
from ..models.modeling_outputs import Transformer2DModelOutput
from ..utils import logging
from ._common import _ATTENTION_CLASSES
from .hooks import HookRegistry, ModelHook


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_FASTER_CACHE_DENOISER_HOOK = "faster_cache_denoiser"
_FASTER_CACHE_BLOCK_HOOK = "faster_cache_block"
_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "^blocks.*attn",
    "^transformer_blocks.*attn",
    "^single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("^temporal_transformer_blocks.*attn",)
_TRANSFORMER_BLOCK_IDENTIFIERS = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS + _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
_UNCOND_COND_INPUT_KWARGS_IDENTIFIERS = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "attention_mask",
    "encoder_attention_mask",
)


@dataclass
class FasterCacheConfig:
    r"""
    Configuration for [FasterCache](https://huggingface.co/papers/2410.19355).

    Attributes:
        spatial_attention_block_skip_range (`int`, defaults to `2`):
            Calculate the attention states every `N` iterations. If this is set to `N`, the attention computation will
            be skipped `N - 1` times (i.e., cached attention states will be reused) before computing the new attention
            states again.
        temporal_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            Calculate the attention states every `N` iterations. If this is set to `N`, the attention computation will
            be skipped `N - 1` times (i.e., cached attention states will be reused) before computing the new attention
            states again.
        spatial_attention_timestep_skip_range (`Tuple[float, float]`, defaults to `(-1, 681)`):
            The timestep range within which the spatial attention computation can be skipped without a significant loss
            in quality. This is to be determined by the user based on the underlying model. The first value in the
            tuple is the lower bound and the second value is the upper bound. Typically, diffusion timesteps for
            denoising are in the reversed range of 0 to 1000 (i.e. denoising starts at timestep 1000 and ends at
            timestep 0). For the default values, this would mean that the spatial attention computation skipping will
            be applicable only after denoising timestep 681 is reached, and continue until the end of the denoising
            process.
        temporal_attention_timestep_skip_range (`Tuple[float, float]`, *optional*, defaults to `None`):
            The timestep range within which the temporal attention computation can be skipped without a significant
            loss in quality. This is to be determined by the user based on the underlying model. The first value in the
            tuple is the lower bound and the second value is the upper bound. Typically, diffusion timesteps for
            denoising are in the reversed range of 0 to 1000 (i.e. denoising starts at timestep 1000 and ends at
            timestep 0).
        low_frequency_weight_update_timestep_range (`Tuple[int, int]`, defaults to `(99, 901)`):
            The timestep range within which the low frequency weight scaling update is applied. The first value in the
            tuple is the lower bound and the second value is the upper bound of the timestep range. The callback
            function for the update is called only within this range.
        high_frequency_weight_update_timestep_range (`Tuple[int, int]`, defaults to `(-1, 301)`):
            The timestep range within which the high frequency weight scaling update is applied. The first value in the
            tuple is the lower bound and the second value is the upper bound of the timestep range. The callback
            function for the update is called only within this range.
        alpha_low_frequency (`float`, defaults to `1.1`):
            The weight to scale the low frequency updates by. This is used to approximate the unconditional branch from
            the conditional branch outputs.
        alpha_high_frequency (`float`, defaults to `1.1`):
            The weight to scale the high frequency updates by. This is used to approximate the unconditional branch
            from the conditional branch outputs.
        unconditional_batch_skip_range (`int`, defaults to `5`):
            Process the unconditional branch every `N` iterations. If this is set to `N`, the unconditional branch
            computation will be skipped `N - 1` times (i.e., cached unconditional branch states will be reused) before
            computing the new unconditional branch states again.
        unconditional_batch_timestep_skip_range (`Tuple[float, float]`, defaults to `(-1, 641)`):
            The timestep range within which the unconditional branch computation can be skipped without a significant
            loss in quality. This is to be determined by the user based on the underlying model. The first value in the
            tuple is the lower bound and the second value is the upper bound.
        spatial_attention_block_identifiers (`Tuple[str, ...]`, defaults to `("blocks.*attn1", "transformer_blocks.*attn1", "single_transformer_blocks.*attn1")`):
            The identifiers to match the spatial attention blocks in the model. If the name of the block contains any
            of these identifiers, FasterCache will be applied to that block. This can either be the full layer names,
            partial layer names, or regex patterns. Matching will always be done using a regex match.
        temporal_attention_block_identifiers (`Tuple[str, ...]`, defaults to `("temporal_transformer_blocks.*attn1",)`):
            The identifiers to match the temporal attention blocks in the model. If the name of the block contains any
            of these identifiers, FasterCache will be applied to that block. This can either be the full layer names,
            partial layer names, or regex patterns. Matching will always be done using a regex match.
        attention_weight_callback (`Callable[[torch.nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the attention outputs by. This function should take
            the attention module as input and return a float value. This is used to approximate the unconditional
            branch from the conditional branch outputs. If not provided, the default weight is 0.5 for all timesteps.
            Typically, as described in the paper, this weight should gradually increase from 0 to 1 as the inference
            progresses. Users are encouraged to experiment and provide custom weight schedules that take into account
            the number of inference steps and underlying model behaviour as denoising progresses.
        low_frequency_weight_callback (`Callable[[torch.nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the low frequency updates by. If not provided, the
            default weight is 1.1 for timesteps within the range specified (as described in the paper).
        high_frequency_weight_callback (`Callable[[torch.nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the high frequency updates by. If not provided, the
            default weight is 1.1 for timesteps within the range specified (as described in the paper).
        tensor_format (`str`, defaults to `"BCFHW"`):
            The format of the input tensors. This should be one of `"BCFHW"`, `"BFCHW"`, or `"BCHW"`. The format is
            used to split individual latent frames in order for low and high frequency components to be computed.
        is_guidance_distilled (`bool`, defaults to `False`):
            Whether the model is guidance distilled or not. If the model is guidance distilled, FasterCache will not be
            applied at the denoiser-level to skip the unconditional branch computation (as there is none).
        _unconditional_conditional_input_kwargs_identifiers (`List[str]`, defaults to `("hidden_states", "encoder_hidden_states", "timestep", "attention_mask", "encoder_attention_mask")`):
            The identifiers to match the input kwargs that contain the batchwise-concatenated unconditional and
            conditional inputs. If the name of the input kwargs contains any of these identifiers, FasterCache will
            split the inputs into unconditional and conditional branches. This must be a list of exact input kwargs
            names that contain the batchwise-concatenated unconditional and conditional inputs.
    """

    # In the paper and codebase, they hardcode these values to 2. However, it can be made configurable
    # after some testing. We default to 2 if these parameters are not provided.
    spatial_attention_block_skip_range: int = 2
    temporal_attention_block_skip_range: Optional[int] = None

    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)

    # Indicator functions for low/high frequency as mentioned in Equation 11 of the paper
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)

    # ⍺1 and ⍺2 as mentioned in Equation 11 of the paper
    alpha_low_frequency: float = 1.1
    alpha_high_frequency: float = 1.1

    # n as described in CFG-Cache explanation in the paper - dependent on the model
    unconditional_batch_skip_range: int = 5
    unconditional_batch_timestep_skip_range: Tuple[int, int] = (-1, 641)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS

    attention_weight_callback: Callable[[torch.nn.Module], float] = None
    low_frequency_weight_callback: Callable[[torch.nn.Module], float] = None
    high_frequency_weight_callback: Callable[[torch.nn.Module], float] = None

    tensor_format: str = "BCFHW"
    is_guidance_distilled: bool = False

    current_timestep_callback: Callable[[], int] = None

    _unconditional_conditional_input_kwargs_identifiers: List[str] = _UNCOND_COND_INPUT_KWARGS_IDENTIFIERS

    def __repr__(self) -> str:
        return (
            f"FasterCacheConfig(\n"
            f"  spatial_attention_block_skip_range={self.spatial_attention_block_skip_range},\n"
            f"  temporal_attention_block_skip_range={self.temporal_attention_block_skip_range},\n"
            f"  spatial_attention_timestep_skip_range={self.spatial_attention_timestep_skip_range},\n"
            f"  temporal_attention_timestep_skip_range={self.temporal_attention_timestep_skip_range},\n"
            f"  low_frequency_weight_update_timestep_range={self.low_frequency_weight_update_timestep_range},\n"
            f"  high_frequency_weight_update_timestep_range={self.high_frequency_weight_update_timestep_range},\n"
            f"  alpha_low_frequency={self.alpha_low_frequency},\n"
            f"  alpha_high_frequency={self.alpha_high_frequency},\n"
            f"  unconditional_batch_skip_range={self.unconditional_batch_skip_range},\n"
            f"  unconditional_batch_timestep_skip_range={self.unconditional_batch_timestep_skip_range},\n"
            f"  spatial_attention_block_identifiers={self.spatial_attention_block_identifiers},\n"
            f"  temporal_attention_block_identifiers={self.temporal_attention_block_identifiers},\n"
            f"  tensor_format={self.tensor_format},\n"
            f")"
        )


class FasterCacheDenoiserState:
    r"""
    State for [FasterCache](https://huggingface.co/papers/2410.19355) top-level denoiser module.
    """

    def __init__(self) -> None:
        self.iteration: int = 0
        self.low_frequency_delta: torch.Tensor = None
        self.high_frequency_delta: torch.Tensor = None

    def reset(self):
        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None


class FasterCacheBlockState:
    r"""
    State for [FasterCache](https://huggingface.co/papers/2410.19355). Every underlying block that FasterCache is
    applied to will have an instance of this state.
    """

    def __init__(self) -> None:
        self.iteration: int = 0
        self.batch_size: int = None
        self.cache: Tuple[torch.Tensor, torch.Tensor] = None

    def reset(self):
        self.iteration = 0
        self.batch_size = None
        self.cache = None


class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        unconditional_batch_skip_range: int,
        unconditional_batch_timestep_skip_range: Tuple[int, int],
        tensor_format: str,
        is_guidance_distilled: bool,
        uncond_cond_input_kwargs_identifiers: List[str],
        current_timestep_callback: Callable[[], int],
        low_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[torch.nn.Module], torch.Tensor],
    ) -> None:
        super().__init__()

        self.unconditional_batch_skip_range = unconditional_batch_skip_range
        self.unconditional_batch_timestep_skip_range = unconditional_batch_timestep_skip_range
        # We can't easily detect what args are to be split in unconditional and conditional branches. We
        # can only do it for kwargs, hence they are the only ones we split. The args are passed as-is.
        # If a model is to be made compatible with FasterCache, the user must ensure that the inputs that
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format
        self.is_guidance_distilled = is_guidance_distilled

        self.current_timestep_callback = current_timestep_callback
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback

    def initialize_hook(self, module):
        self.state = FasterCacheDenoiserState()
        return module

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input tensor is batchwise-concatenated with unconditional inputs
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        # Split the unconditional and conditional inputs. We only want to infer the conditional branch if the
        # requirements for skipping the unconditional branch are met as described in the paper.
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the range specified by the user. This is the optimal timestep range
        #      where approximating the unconditional branch from the computation of the conditional branch is possible
        #      without a significant loss in quality.
        #   3. The current iteration is not a multiple of the unconditional batch skip range. This is done so that
        #      we compute the unconditional branch at least once every few iterations to ensure minimal quality loss.
        is_within_timestep_range = (
            self.unconditional_batch_timestep_skip_range[0]
            < self.current_timestep_callback()
            < self.unconditional_batch_timestep_skip_range[1]
        )
        should_skip_uncond = (
            self.state.iteration > 0
            and is_within_timestep_range
            and self.state.iteration % self.unconditional_batch_skip_range != 0
            and not self.is_guidance_distilled
        )

        if should_skip_uncond:
            is_any_kwarg_uncond = any(k in self.uncond_cond_input_kwargs_identifiers for k in kwargs.keys())
            if is_any_kwarg_uncond:
                logger.debug("FasterCache - Skipping unconditional branch computation")
                args = tuple([self._get_cond_input(arg) if torch.is_tensor(arg) else arg for arg in args])
                kwargs = {
                    k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                    for k, v in kwargs.items()
                }

        output = self.fn_ref.original_forward(*args, **kwargs)

        if self.is_guidance_distilled:
            self.state.iteration += 1
            return output

        if torch.is_tensor(output):
            hidden_states = output
        elif isinstance(output, (tuple, Transformer2DModelOutput)):
            hidden_states = output[0]

        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            self.state.low_frequency_delta = self.state.low_frequency_delta * self.low_frequency_weight_callback(
                module
            )
            self.state.high_frequency_delta = self.state.high_frequency_delta * self.high_frequency_weight_callback(
                module
            )

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditional branch outputs as described in Equation 9 and 10 of the paper
            low_freq_uncond = self.state.low_frequency_delta + low_freq_cond
            high_freq_uncond = self.state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = torch.fft.ifftshift(uncond_freq)
            uncond_states = torch.fft.ifft2(uncond_states).real

            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.unflatten(0, (batch_size, -1))
                hidden_states = hidden_states.unflatten(0, (batch_size, -1))
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

            # Concatenate the approximated unconditional and predicted conditional branches
            uncond_states = uncond_states.to(hidden_states.dtype)
            hidden_states = torch.cat([uncond_states, hidden_states], dim=0)
        else:
            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            self.state.low_frequency_delta = low_freq_uncond - low_freq_cond
            self.state.high_frequency_delta = high_freq_uncond - high_freq_cond

        self.state.iteration += 1
        if torch.is_tensor(output):
            output = hidden_states
        elif isinstance(output, tuple):
            output = (hidden_states, *output[1:])
        else:
            output.sample = hidden_states

        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module


class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        block_skip_range: int,
        timestep_skip_range: Tuple[int, int],
        is_guidance_distilled: bool,
        weight_callback: Callable[[torch.nn.Module], float],
        current_timestep_callback: Callable[[], int],
    ) -> None:
        super().__init__()

        self.block_skip_range = block_skip_range
        self.timestep_skip_range = timestep_skip_range
        self.is_guidance_distilled = is_guidance_distilled

        self.weight_callback = weight_callback
        self.current_timestep_callback = current_timestep_callback

    def initialize_hook(self, module):
        self.state = FasterCacheBlockState()
        return module

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        if t_2_output.size(0) != batch_size:
            # The cache t_2_output contains both batchwise-concatenated unconditional-conditional branch outputs. Just
            # take the conditional branch outputs.
            assert t_2_output.size(0) == 2 * batch_size
            t_2_output = t_2_output[batch_size:]
        if t_output.size(0) != batch_size:
            # The cache t_output contains both batchwise-concatenated unconditional-conditional branch outputs. Just
            # take the conditional branch outputs.
            assert t_output.size(0) == 2 * batch_size
            t_output = t_output[batch_size:]
        return t_output + (t_output - t_2_output) * weight

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if self.state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            self.state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants to infer both unconditional and conditional branches. This
        # is because the expected output shapes of attention layer will not match if we only return values from
        # the cache (which only caches conditional branch outputs). So, if state.batch_size (which is the true
        # unconditional-conditional batch size) is same as the current batch size, we don't perform the layer
        # skip. Otherwise, we conditionally skip the layer based on what state.skip_callback returns.
        is_within_timestep_range = (
            self.timestep_skip_range[0] < self.current_timestep_callback() < self.timestep_skip_range[1]
        )
        if not is_within_timestep_range:
            should_skip_attention = False
        else:
            should_compute_attention = self.state.iteration > 0 and self.state.iteration % self.block_skip_range == 0
            should_skip_attention = not should_compute_attention
        if should_skip_attention:
            should_skip_attention = self.is_guidance_distilled or self.state.batch_size != batch_size

        if should_skip_attention:
            logger.debug("FasterCache - Skipping attention and using approximation")
            if torch.is_tensor(self.state.cache[-1]):
                t_2_output, t_output = self.state.cache
                weight = self.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tensors from past N iterations (N=2 for FasterCache). We need to handle all of them.
                # Diffusers blocks can return multiple tensors - let's call them [A, B, C, ...] for simplicity.
                # In our cache, we would have [[A_1, B_1, C_1, ...], [A_2, B_2, C_2, ...], ...] where each list is the output from
                # a forward pass of the block. We need to compute the approximated output for each of these tensors.
                # The zip(*state.cache) operation will give us [(A_1, A_2, ...), (B_1, B_2, ...), (C_1, C_2, ...), ...] which
                # allows us to compute the approximated attention output for each tensor in the cache.
                output = ()
                for t_2_output, t_output in zip(*self.state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, self.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("FasterCache - Computing attention")
            output = self.fn_ref.original_forward(*args, **kwargs)

        # Note that the following condition for getting hidden_states should suffice since Diffusers blocks either return
        # a single hidden_states tensor, or a tuple of (hidden_states, encoder_hidden_states) tensors. We need to handle
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not self.is_guidance_distilled and cache_output.size(0) == self.state.batch_size:
                # The output here can be both unconditional-conditional branch outputs or just conditional branch outputs.
                # This is determined at the higher-level denoiser module. We only want to cache the conditional branch outputs.
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not self.is_guidance_distilled and out.size(0) == self.state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if self.state.cache is None:
            self.state.cache = [cache_output, cache_output]
        else:
            self.state.cache = [self.state.cache[-1], cache_output]

        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> torch.nn.Module:
        self.state.reset()
        return module


def apply_faster_cache(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    r"""
    Applies [FasterCache](https://huggingface.co/papers/2410.19355) to a given pipeline.

    Args:
        module (`torch.nn.Module`):
            The pytorch module to apply FasterCache to. Typically, this should be a transformer architecture supported
            in Diffusers, such as `CogVideoXTransformer3DModel`, but external implementations may also work.
        config (`FasterCacheConfig`):
            The configuration to use for FasterCache.

    Example:
    ```python
    >>> import torch
    >>> from diffusers import CogVideoXPipeline, FasterCacheConfig, apply_faster_cache

    >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> config = FasterCacheConfig(
    ...     spatial_attention_block_skip_range=2,
    ...     spatial_attention_timestep_skip_range=(-1, 681),
    ...     low_frequency_weight_update_timestep_range=(99, 641),
    ...     high_frequency_weight_update_timestep_range=(-1, 301),
    ...     spatial_attention_block_identifiers=["transformer_blocks"],
    ...     attention_weight_callback=lambda _: 0.3,
    ...     tensor_format="BFCHW",
    ... )
    >>> apply_faster_cache(pipe.transformer, config)
    ```
    """

    logger.warning(
        "FasterCache is a purely experimental feature and may not work as expected. Not all models support FasterCache. "
        "The API is subject to change in future releases, with no guarantee of backward compatibility. Please report any issues at "
        "https://github.com/huggingface/diffusers/issues."
    )

    if config.attention_weight_callback is None:
        # If the user has not provided a weight callback, we default to 0.5 for all timesteps.
        # In the paper, they recommend using a gradually increasing weight from 0 to 1 as the inference progresses, but
        # this depends from model-to-model. It is required by the user to provide a weight callback if they want to
        # use a different weight function. Defaulting to 0.5 works well in practice for most cases.
        logger.warning(
            "No `attention_weight_callback` provided when enabling FasterCache. Defaulting to using a weight of 0.5 for all timesteps."
        )
        config.attention_weight_callback = lambda _: 0.5

    if config.low_frequency_weight_callback is None:
        logger.debug(
            "Low frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )

        def low_frequency_weight_callback(module: torch.nn.Module) -> float:
            is_within_range = (
                config.low_frequency_weight_update_timestep_range[0]
                < config.current_timestep_callback()
                < config.low_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_low_frequency if is_within_range else 1.0

        config.low_frequency_weight_callback = low_frequency_weight_callback

    if config.high_frequency_weight_callback is None:
        logger.debug(
            "High frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )

        def high_frequency_weight_callback(module: torch.nn.Module) -> float:
            is_within_range = (
                config.high_frequency_weight_update_timestep_range[0]
                < config.current_timestep_callback()
                < config.high_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_high_frequency if is_within_range else 1.0

        config.high_frequency_weight_callback = high_frequency_weight_callback

    supported_tensor_formats = ["BCFHW", "BFCHW", "BCHW"]  # TODO(aryan): Support BSC for LTX Video
    if config.tensor_format not in supported_tensor_formats:
        raise ValueError(f"`tensor_format` must be one of {supported_tensor_formats}, but got {config.tensor_format}.")

    _apply_faster_cache_on_denoiser(module, config)

    for name, submodule in module.named_modules():
        if not isinstance(submodule, _ATTENTION_CLASSES):
            continue
        if any(re.search(identifier, name) is not None for identifier in _TRANSFORMER_BLOCK_IDENTIFIERS):
            _apply_faster_cache_on_attention_class(name, submodule, config)


def _apply_faster_cache_on_denoiser(module: torch.nn.Module, config: FasterCacheConfig) -> None:
    hook = FasterCacheDenoiserHook(
        config.unconditional_batch_skip_range,
        config.unconditional_batch_timestep_skip_range,
        config.tensor_format,
        config.is_guidance_distilled,
        config._unconditional_conditional_input_kwargs_identifiers,
        config.current_timestep_callback,
        config.low_frequency_weight_callback,
        config.high_frequency_weight_callback,
    )
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, _FASTER_CACHE_DENOISER_HOOK)


def _apply_faster_cache_on_attention_class(name: str, module: AttentionModuleMixin, config: FasterCacheConfig) -> None:
    is_spatial_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.spatial_attention_block_identifiers)
        and config.spatial_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_temporal_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.temporal_attention_block_identifiers)
        and config.temporal_attention_block_skip_range is not None
        and not module.is_cross_attention
    )

    block_skip_range, timestep_skip_range, block_type = None, None, None
    if is_spatial_self_attention:
        block_skip_range = config.spatial_attention_block_skip_range
        timestep_skip_range = config.spatial_attention_timestep_skip_range
        block_type = "spatial"
    elif is_temporal_self_attention:
        block_skip_range = config.temporal_attention_block_skip_range
        timestep_skip_range = config.temporal_attention_timestep_skip_range
        block_type = "temporal"

    if block_skip_range is None or timestep_skip_range is None:
        logger.debug(
            f'Unable to apply FasterCache to the selected layer: "{name}" because it does '
            f"not match any of the required criteria for spatial or temporal attention layers. Note, "
            f"however, that this layer may still be valid for applying PAB. Please specify the correct "
            f"block identifiers in the configuration or use the specialized `apply_faster_cache_on_module` "
            f"function to apply FasterCache to this layer."
        )
        return

    logger.debug(f"Enabling FasterCache ({block_type}) for layer: {name}")
    hook = FasterCacheBlockHook(
        block_skip_range,
        timestep_skip_range,
        config.is_guidance_distilled,
        config.attention_weight_callback,
        config.current_timestep_callback,
    )
    registry = HookRegistry.check_if_exists_or_initialize(module)
    registry.register_hook(hook, _FASTER_CACHE_BLOCK_HOOK)


# Reference: https://github.com/Vchitect/FasterCache/blob/fab32c15014636dc854948319c0a9a8d92c7acb4/scripts/latte/faster_cache_sample_latte.py#L127C1-L143C39
@torch.no_grad()
def _split_low_high_freq(x):
    fft = torch.fft.fft2(x)
    fft_shifted = torch.fft.fftshift(fft)
    height, width = x.shape[-2:]
    radius = min(height, width) // 5

    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_x, center_y = width // 2, height // 2
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(x.device)
    high_freq_mask = ~low_freq_mask

    low_freq_fft = fft_shifted * low_freq_mask
    high_freq_fft = fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft
