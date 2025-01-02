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

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.fft as FFT
import torch.nn as nn

from ..models.attention_processor import Attention
from ..models.hooks import ModelHook, add_hook_to_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO(aryan): handle mochi attention
_ATTENTION_CLASSES = (Attention,)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "blocks.*attn",
    "transformer_blocks.*attn",
    "single_transformer_blocks.*attn",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks.*attn",)
_UNCOND_COND_INPUT_KWARGS_IDENTIFIERS = (
    "hidden_states",
    "encoder_hidden_states",
    "timestep",
    "attention_mask",
    "encoder_attention_mask",
)
_GUIDANCE_DISTILLATION_KWARGS_IDENTIFIERS = ("guidance",)


@dataclass
class FasterCacheConfig:
    r"""
    Configuration for [FasterCache](https://huggingface.co/papers/2410.19355).

    Attributes:
        spatial_attention_block_skip_range (`int`, defaults to `2`):
            Calculate the attention states every `N` iterations. If this is set to `N`, the attention computation will
            be skipped `N - 1` times (i.e., cached attention states will be re-used) before computing the new attention
            states again.
        temporal_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            Calculate the attention states every `N` iterations. If this is set to `N`, the attention computation will
            be skipped `N - 1` times (i.e., cached attention states will be re-used) before computing the new attention
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
            computation will be skipped `N - 1` times (i.e., cached unconditional branch states will be re-used) before
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
        attention_weight_callback (`Callable[[nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the attention outputs by. This function should take
            the attention module as input and return a float value. This is used to approximate the unconditional
            branch from the conditional branch outputs. If not provided, the default weight is 0.5 for all timesteps.
            Typically, as described in the paper, this weight should gradually increase from 0 to 1 as the inference
            progresses. Users are encouraged to experiment and provide custom weight schedules that take into account
            the number of inference steps and underlying model behaviour as denoising progresses.
        low_frequency_weight_callback (`Callable[[nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the low frequency updates by. If not provided, the
            default weight is 1.1 for timesteps within the range specified (as described in the paper).
        high_frequency_weight_callback (`Callable[[nn.Module], float]`, defaults to `None`):
            The callback function to determine the weight to scale the high frequency updates by. If not provided, the
            default weight is 1.1 for timesteps within the range specified (as described in the paper).
        tensor_format (`str`, defaults to `"BCFHW"`):
            The format of the input tensors. This should be one of `"BCFHW"`, `"BFCHW"`, or `"BCHW"`. The format is
            used to split individual latent frames in order for low and high frequency components to be computed.
        _unconditional_conditional_input_kwargs_identifiers (`List[str]`, defaults to `("hidden_states", "encoder_hidden_states", "timestep", "attention_mask", "encoder_attention_mask")`):
            The identifiers to match the input kwargs that contain the batchwise-concatenated unconditional and
            conditional inputs. If the name of the input kwargs contains any of these identifiers, FasterCache will
            split the inputs into unconditional and conditional branches. This must be a list of exact input kwargs
            names that contain the batchwise-concatenated unconditional and conditional inputs.
        _guidance_distillation_kwargs_identifiers (`List[str]`, defaults to `("guidance",)`):
            The identifiers to match the input kwargs that contain the guidance distillation inputs. If the name of the
            input kwargs contains any of these identifiers, FasterCache will not split the inputs into unconditional
            and conditional branches (unconditional branches are only computed sometimes based on certain checks). This
            allows usage of FasterCache in models like Flux-Dev and HunyuanVideo which are guidance-distilled (only
            attention skipping related parts are applied, and not unconditional branch approximation).
    """

    # In the paper and codebase, they hardcode these values to 2. However, it can be made configurable
    # after some testing. We default to 2 if these parameters are not provided.
    spatial_attention_block_skip_range: int = 2
    temporal_attention_block_skip_range: Optional[int] = None

    # TODO(aryan): write heuristics for what the best way to obtain these values are
    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)

    # Indicator functions for low/high frequency as mentioned in Equation 11 of the paper
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)

    # ⍺1 and ⍺2 as mentioned in Equation 11 of the paper
    alpha_low_frequency: float = 1.1
    alpha_high_frequency: float = 1.1

    # n as described in CFG-Cache explanation in the paper - dependant on the model
    unconditional_batch_skip_range: int = 5
    unconditional_batch_timestep_skip_range: Tuple[int, int] = (-1, 641)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS

    attention_weight_callback: Callable[[nn.Module], float] = None
    low_frequency_weight_callback: Callable[[nn.Module], float] = None
    high_frequency_weight_callback: Callable[[nn.Module], float] = None

    tensor_format: str = "BCFHW"

    _unconditional_conditional_input_kwargs_identifiers: List[str] = _UNCOND_COND_INPUT_KWARGS_IDENTIFIERS
    _guidance_distillation_kwargs_identifiers: List[str] = _GUIDANCE_DISTILLATION_KWARGS_IDENTIFIERS


class FasterCacheDenoiserState:
    r"""
    State for [FasterCache](https://huggingface.co/papers/2410.19355) top-level denoiser module.
    """

    def __init__(
        self,
        low_frequency_weight_callback: Callable[[nn.Module], torch.Tensor],
        high_frequency_weight_callback: Callable[[nn.Module], torch.Tensor],
        uncond_skip_callback: Callable[[nn.Module], bool],
    ) -> None:
        self.low_frequency_weight_callback = low_frequency_weight_callback
        self.high_frequency_weight_callback = high_frequency_weight_callback
        self.uncond_skip_callback = uncond_skip_callback

        self.iteration: int = 0
        self.low_frequency_delta: torch.Tensor = None
        self.high_frequency_delta: torch.Tensor = None
        self.is_guidance_distilled: bool = None

    def reset(self):
        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None
        self.is_guidance_distilled = None


class FasterCacheBlockState:
    r"""
    State for [FasterCache](https://huggingface.co/papers/2410.19355). Every underlying block that FasterCache is
    applied to will have an instance of this state.

    Attributes:
        iteration (`int`):
            The current iteration of the FasterCache. It is necessary to ensure that `reset_state` is called before
            starting a new inference forward pass for this to work correctly.
    """

    def __init__(
        self, skip_callback: Callable[[nn.Module], bool], weight_callback: Callable[[nn.Module], float]
    ) -> None:
        self.skip_callback = skip_callback
        self.weight_callback = weight_callback

        self.iteration: int = 0
        self.batch_size: int = None
        self.cache: Tuple[torch.Tensor, torch.Tensor] = None
        self.is_guidance_distilled: bool = None

    def reset(self):
        self.iteration = 0
        self.batch_size = None
        self.cache = None
        self.is_guidance_distilled = None


def apply_faster_cache(
    pipeline: DiffusionPipeline,
    config: Optional[FasterCacheConfig] = None,
) -> None:
    r"""
    Applies [FasterCache](https://huggingface.co/papers/2410.19355) to a given pipeline.

    Args:
        pipeline (`DiffusionPipeline`):
            The diffusion pipeline to apply FasterCache to.
        config (`Optional[FasterCacheConfig]`, `optional`, defaults to `None`):
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
    >>> apply_faster_cache(pipe, config)
    ```
    """

    if config is None:
        logger.warning("No FasterCacheConfig provided. Using default configuration.")
        config = FasterCacheConfig()

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

        def low_frequency_weight_callback(module: nn.Module) -> float:
            is_within_range = (
                config.low_frequency_weight_update_timestep_range[0]
                < pipeline._current_timestep
                < config.low_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_low_frequency if is_within_range else 1.0

        config.low_frequency_weight_callback = low_frequency_weight_callback

    if config.high_frequency_weight_callback is None:
        logger.debug(
            "High frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )

        def high_frequency_weight_callback(module: nn.Module) -> float:
            is_within_range = (
                config.high_frequency_weight_update_timestep_range[0]
                < pipeline._current_timestep
                < config.high_frequency_weight_update_timestep_range[1]
            )
            return config.alpha_high_frequency if is_within_range else 1.0

        config.high_frequency_weight_callback = high_frequency_weight_callback

    supported_tensor_formats = ["BCFHW", "BFCHW", "BCHW"]  # TODO(aryan): Support BSC for LTX Video
    if config.tensor_format not in supported_tensor_formats:
        raise ValueError(f"`tensor_format` must be one of {supported_tensor_formats}, but got {config.tensor_format}.")

    denoiser = pipeline.transformer if hasattr(pipeline, "transformer") else pipeline.unet
    _apply_fastercache_on_denoiser(pipeline, denoiser, config)

    for name, module in denoiser.named_modules():
        if not isinstance(module, _ATTENTION_CLASSES):
            continue
        if isinstance(module, Attention):
            _apply_fastercache_on_attention_class(pipeline, name, module, config)


def _apply_fastercache_on_denoiser(
    pipeline: DiffusionPipeline, denoiser: nn.Module, config: FasterCacheConfig
) -> None:
    def uncond_skip_callback(module: nn.Module) -> bool:
        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the range specified by the user. This is the optimal timestep range
        #      where approximating the unconditional branch from the computation of the conditional branch is possible
        #      without a significant loss in quality.
        #   3. The current iteration is not a multiple of the unconditional batch skip range. This is done so that
        #      we compute the unconditional branch at least once every few iterations to ensure minimal quality loss.

        state: FasterCacheDenoiserState = module._fastercache_state
        is_within_range = (
            config.unconditional_batch_timestep_skip_range[0]
            < pipeline._current_timestep
            < config.unconditional_batch_timestep_skip_range[1]
        )
        return state.iteration > 0 and is_within_range and state.iteration % config.unconditional_batch_skip_range != 0

    denoiser._fastercache_state = FasterCacheDenoiserState(
        config.low_frequency_weight_callback, config.high_frequency_weight_callback, uncond_skip_callback
    )
    hook = FasterCacheDenoiserHook(
        config._unconditional_conditional_input_kwargs_identifiers,
        config._guidance_distillation_kwargs_identifiers,
        config.tensor_format,
    )
    add_hook_to_module(denoiser, hook, append=True)


def _apply_fastercache_on_attention_class(
    pipeline: DiffusionPipeline, name: str, module: Attention, config: FasterCacheConfig
) -> None:
    is_spatial_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.spatial_attention_block_identifiers)
        and config.spatial_attention_block_skip_range is not None
        and not module.is_cross_attention
    )
    is_temporal_self_attention = (
        any(
            f"{identifier}." in name or identifier == name
            for identifier in config.temporal_attention_block_identifiers
        )
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
            f"block identifiers in the configuration or use the specialized `apply_fastercache_on_module` "
            f"function to apply FasterCache to this layer."
        )
        return

    def skip_callback(module: nn.Module) -> bool:
        fastercache_state: FasterCacheBlockState = module._fastercache_state
        is_within_timestep_range = timestep_skip_range[0] < pipeline._current_timestep < timestep_skip_range[1]

        if not is_within_timestep_range:
            # We are still not in the phase of inference where skipping attention is possible without minimal quality
            # loss, as described in the paper. So, the attention computation cannot be skipped
            return False

        should_compute_attention = (
            fastercache_state.iteration > 0 and fastercache_state.iteration % block_skip_range == 0
        )
        return not should_compute_attention

    logger.debug(f"Enabling FasterCache ({block_type}) for layer: {name}")
    module._fastercache_state = FasterCacheBlockState(skip_callback, config.attention_weight_callback)
    hook = FasterCacheBlockHook()
    add_hook_to_module(module, hook, append=True)


class FasterCacheDenoiserHook(ModelHook):
    _is_stateful = True

    def __init__(
        self,
        uncond_cond_input_kwargs_identifiers: List[str],
        guidance_distillation_kwargs_identifiers: List[str],
        tensor_format: str,
    ) -> None:
        super().__init__()

        # We can't easily detect what args are to be split in unconditional and conditional branches. We
        # can only do it for kwargs, hence they are the only ones we split. The args are passed as-is.
        # If a model is to be made compatible with FasterCache, the user must ensure that the inputs that
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers

        # See documentation for `guidance_distillation_kwargs_identifiers` in FasterCacheConfig for more information
        self.guidance_distillation_kwargs_identifiers = guidance_distillation_kwargs_identifiers

        self.tensor_format = tensor_format

    @staticmethod
    def _get_cond_input(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Note: this method assumes that the input tensor is batchwise-concatenated with unconditional inputs
        # followed by conditional inputs.
        _, cond = input.chunk(2, dim=0)
        return cond

    def new_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)
        state: FasterCacheDenoiserState = module._fastercache_state

        # Split the unconditional and conditional inputs. We only want to infer the conditional branch if the
        # requirements for skipping the unconditional branch are met as described in the paper.
        should_skip_uncond = state.uncond_skip_callback(module)

        if state.is_guidance_distilled is None:
            # The following check assumes that the guidance embedding are torch tensors for check to pass. This
            # seems to be true for all models supported in diffusers
            state.is_guidance_distilled = any(
                identifier in kwargs and kwargs[identifier] is not None and torch.is_tensor(kwargs[identifier])
                for identifier in self.guidance_distillation_kwargs_identifiers
            )
            # Make all children FasterCacheBlockHooks aware of whether the model is guidance distilled or not
            # because we cannot determine this within the block hooks
            for name, child_module in module.named_modules():
                if hasattr(child_module, "_fastercache_state") and isinstance(
                    child_module._fastercache_state, FasterCacheBlockState
                ):
                    # TODO(aryan): remove later
                    logger.debug(f"Setting guidance distillation flag for layer: {name}")
                    child_module._fastercache_state.is_guidance_distilled = state.is_guidance_distilled
        assert state.is_guidance_distilled is not None

        if should_skip_uncond and not state.is_guidance_distilled:
            kwargs = {
                k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                for k, v in kwargs.items()
            }
            # TODO(aryan): remove later
            logger.debug("Skipping unconditional branch computation")

        output = module._old_forward(*args, **kwargs)

        if state.is_guidance_distilled:
            state.iteration += 1
            return output

        # TODO(aryan): handle Transformer2DModelOutput
        hidden_states = output[0] if isinstance(output, tuple) else output
        batch_size = hidden_states.size(0)

        if should_skip_uncond:
            state.low_frequency_delta = state.low_frequency_delta * state.low_frequency_weight_callback(module)
            state.high_frequency_delta = state.high_frequency_delta * state.high_frequency_weight_callback(module)

            if self.tensor_format == "BCFHW":
                hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                hidden_states = hidden_states.flatten(0, 1)

            low_freq_cond, high_freq_cond = _split_low_high_freq(hidden_states.float())

            # Approximate/compute the unconditional branch outputs as described in Equation 9 and 10 of the paper
            low_freq_uncond = state.low_frequency_delta + low_freq_cond
            high_freq_uncond = state.high_frequency_delta + high_freq_cond
            uncond_freq = low_freq_uncond + high_freq_uncond

            uncond_states = FFT.ifftshift(uncond_freq)
            uncond_states = FFT.ifft2(uncond_states).real

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
            # TODO(aryan): remove later
            logger.debug("Computing unconditional branch")

            uncond_states, cond_states = hidden_states.chunk(2, dim=0)
            if self.tensor_format == "BCFHW":
                uncond_states = uncond_states.permute(0, 2, 1, 3, 4)
                cond_states = cond_states.permute(0, 2, 1, 3, 4)
            if self.tensor_format == "BCFHW" or self.tensor_format == "BFCHW":
                uncond_states = uncond_states.flatten(0, 1)
                cond_states = cond_states.flatten(0, 1)

            low_freq_uncond, high_freq_uncond = _split_low_high_freq(uncond_states.float())
            low_freq_cond, high_freq_cond = _split_low_high_freq(cond_states.float())
            state.low_frequency_delta = low_freq_uncond - low_freq_cond
            state.high_frequency_delta = high_freq_uncond - high_freq_cond

        state.iteration += 1
        output = (hidden_states, *output[1:]) if isinstance(output, tuple) else hidden_states
        return output

    def reset_state(self, module: nn.Module) -> nn.Module:
        module._fastercache_state.reset()
        return module


class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def _compute_approximated_attention_output(
        self, t_2_output: torch.Tensor, t_output: torch.Tensor, weight: float, batch_size: int
    ) -> torch.Tensor:
        # TODO(aryan): these conditions may not be needed after latest refactor. they exist for safety. do test if they can be removed
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

    def new_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)
        state: FasterCacheBlockState = module._fastercache_state

        # The denoiser should have set this flag for all children FasterCacheBlockHooks to either True or False
        assert state.is_guidance_distilled is not None

        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants to infer both unconditional and conditional branches. This
        # is because the expected output shapes of attention layer will not match if we only return values from
        # the cache (which only caches conditional branch outputs). So, if state.batch_size (which is the true
        # unconditional-conditional batch size) is same as the current batch size, we don't perform the layer
        # skip. Otherwise, we conditionally skip the layer based on what state.skip_callback returns.
        should_skip_attention = state.skip_callback(module) and (
            state.is_guidance_distilled or state.batch_size != batch_size
        )

        if should_skip_attention:
            # TODO(aryan): remove later
            logger.debug("Skipping attention")

            if torch.is_tensor(state.cache[-1]):
                t_2_output, t_output = state.cache
                weight = state.weight_callback(module)
                output = self._compute_approximated_attention_output(t_2_output, t_output, weight, batch_size)
            else:
                # The cache contains multiple tensors from past N iterations (N=2 for FasterCache). We need to handle all of them.
                # Diffusers blocks can return multiple tensors - let's call them [A, B, C, ...] for simplicity.
                # In our cache, we would have [[A_1, B_1, C_1, ...], [A_2, B_2, C_2, ...], ...] where each list is the output from
                # a forward pass of the block. We need to compute the approximated output for each of these tensors.
                # The zip(*state.cache) operation will give us [(A_1, A_2, ...), (B_1, B_2, ...), (C_1, C_2, ...), ...] which
                # allows us to compute the approximated attention output for each tensor in the cache.
                output = ()
                for t_2_output, t_output in zip(*state.cache):
                    result = self._compute_approximated_attention_output(
                        t_2_output, t_output, state.weight_callback(module), batch_size
                    )
                    output += (result,)
        else:
            logger.debug("Computing attention")
            output = module._old_forward(*args, **kwargs)

        # Note that the following condition for getting hidden_states should suffice since Diffusers blocks either return
        # a single hidden_states tensor, or a tuple of (hidden_states, encoder_hidden_states) tensors. We need to handle
        # both cases.
        if torch.is_tensor(output):
            cache_output = output
            if not state.is_guidance_distilled and cache_output.size(0) == state.batch_size:
                # The output here can be both unconditional-conditional branch outputs or just conditional branch outputs.
                # This is determined at the higher-level denoiser module. We only want to cache the conditional branch outputs.
                cache_output = cache_output.chunk(2, dim=0)[1]
        else:
            # Cache all return values and perform the same operation as above
            cache_output = ()
            for out in output:
                if not state.is_guidance_distilled and out.size(0) == state.batch_size:
                    out = out.chunk(2, dim=0)[1]
                cache_output += (out,)

        if state.cache is None:
            state.cache = [cache_output, cache_output]
        else:
            state.cache = [state.cache[-1], cache_output]

        state.iteration += 1
        return module._diffusers_hook.post_forward(module, output)

    def reset_state(self, module: nn.Module) -> nn.Module:
        module._fastercache_state.reset()
        return module


# Reference: https://github.com/Vchitect/FasterCache/blob/fab32c15014636dc854948319c0a9a8d92c7acb4/scripts/latte/fastercache_sample_latte.py#L127C1-L143C39
@torch.no_grad()
def _split_low_high_freq(x):
    fft = FFT.fft2(x)
    fft_shifted = FFT.fftshift(fft)
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
