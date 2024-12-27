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


_ATTENTION_CLASSES = (Attention,)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = (
    "blocks",
    "transformer_blocks",
)
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks",)
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
    """

    num_train_timesteps: int = 1000

    # In the paper and codebase, they hardcode these values to 2. However, it can be made configurable
    # after some testing. We default to 2 if these parameters are not provided.
    spatial_attention_block_skip_range: Optional[int] = None
    temporal_attention_block_skip_range: Optional[int] = None

    # TODO(aryan): write heuristics for what the best way to obtain these values are
    spatial_attention_timestep_skip_range: Tuple[float, float] = (-1, 681)
    temporal_attention_timestep_skip_range: Tuple[float, float] = (-1, 681)

    # Indicator functions for low/high frequency as mentioned in Equation 11 of the paper
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 901)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)

    # ⍺1 and ⍺2 as mentioned in Equation 11 of the paper
    alpha_low_frequency = 1.1
    alpha_high_frequency = 1.1

    # n as described in CFG-Cache explanation in the paper - dependant on the model
    unconditional_batch_skip_range: int = 5
    unconditional_batch_timestep_skip_range: Tuple[float, float] = (-1, 641)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS

    attention_weight_callback: Callable[[nn.Module], float] = None
    low_frequency_weight_callback: Callable[[nn.Module], float] = None
    high_frequency_weight_callback: Callable[[nn.Module], float] = None

    tensor_format: str = "BCFHW"
    unconditional_conditional_input_kwargs_identifiers: List[str] = _UNCOND_COND_INPUT_KWARGS_IDENTIFIERS


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

        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None

    def reset(self):
        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None


class FasterCacheState:
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

        self.iteration = 0
        self.batch_size = None
        self.cache = None

    def reset(self):
        self.iteration = 0
        self.batch_size = None
        self.cache = None


def apply_faster_cache(
    pipeline: DiffusionPipeline,
    config: Optional[FasterCacheConfig] = None,
    denoiser: Optional[nn.Module] = None,
) -> None:
    r"""
    Applies [FasterCache](https://huggingface.co/papers/2410.19355) to a given pipeline.

    Args:
        pipeline (`DiffusionPipeline`):
            The diffusion pipeline to apply FasterCache to.
        config (`Optional[FasterCacheConfig]`, `optional`, defaults to `None`):
            The configuration to use for FasterCache.
        denoiser (`Optional[nn.Module]`, `optional`, defaults to `None`):
            The denoiser module to apply FasterCache to. If `None`, the pipeline's transformer or unet module will be
            used.

    Example:
    ```python
    # TODO(aryan)
    ```
    """

    if config is None:
        config = FasterCacheConfig()

    if config.spatial_attention_block_skip_range is None and config.temporal_attention_block_skip_range is None:
        logger.warning(
            "FasterCache requires one of `spatial_attention_block_skip_range` and/or `temporal_attention_block_skip_range` "
            "to be set to an integer, not `None`. Defaulting to using `spatial_attention_block_skip_range=2` and "
            "`temporal_attention_block_skip_range=2`. To avoid this warning, please set one of the above parameters."
        )
        config.spatial_attention_block_skip_range = 2
        config.temporal_attention_block_skip_range = 2

    if config.attention_weight_callback is None:
        # If the user has not provided a weight callback, we default to 0.5 for all timesteps.
        # In the paper, they recommend using a gradually increasing weight from 0 to 1 as the inference progresses, but
        # this depends from model-to-model. It is required by the user to provide a weight callback if they want to
        # use a different weight function. Defaulting to 0.5 works well in practice for most cases.
        logger.warning(
            "FasterCache requires an `attention_weight_callback` to be set. Defaulting to using a weight of 0.5 for all timesteps."
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

    if denoiser is None:
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
        # If we are not using classifier-free guidance, we cannot skip the denoiser computation. We only compute the
        # conditional branch in this case.
        is_using_classifier_free_guidance = pipeline.do_classifier_free_guidance
        if not is_using_classifier_free_guidance:
            return False

        # We skip the unconditional branch only if the following conditions are met:
        #   1. We have completed at least one iteration of the denoiser
        #   2. The current timestep is within the range specified by the user. This is the optimal timestep range
        #      where approximating the unconditional branch from the computation of the conditional branch is possible
        #      without a significant loss in quality.
        #   3. The current iteration is not a multiple of the unconditional batch skip range. This is to ensure that
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
    hook = FasterCacheModelHook(config.unconditional_conditional_input_kwargs_identifiers, config.tensor_format)
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
        is_using_classifier_free_guidance = pipeline.do_classifier_free_guidance
        if not is_using_classifier_free_guidance:
            return False

        fastercache_state: FasterCacheState = module._fastercache_state
        is_within_timestep_range = timestep_skip_range[0] < pipeline._current_timestep < timestep_skip_range[1]

        if not is_within_timestep_range:
            # We are still not in the phase of inference where skipping attention is possible without minimal quality
            # loss, as described in the paper. So, the attention computation cannot be skipped
            return False
        if fastercache_state.cache is None or fastercache_state.iteration < 2:
            # We need at least 2 iterations to start skipping attention computation
            return False

        should_compute_attention = (
            fastercache_state.iteration > 0 and fastercache_state.iteration % block_skip_range == 0
        )
        return not should_compute_attention

    logger.debug(f"Enabling FasterCache ({block_type}) for layer: {name}")
    module._fastercache_state = FasterCacheState(skip_callback, config.attention_weight_callback)
    hook = FasterCacheBlockHook()
    add_hook_to_module(module, hook, append=True)


class FasterCacheModelHook(ModelHook):
    _is_stateful = True

    def __init__(self, uncond_cond_input_kwargs_identifiers: List[str], tensor_format: str) -> None:
        super().__init__()

        # We can't easily detect what args are to be split in unconditional and conditional branches. We
        # can only do it for kwargs, hence they are the only ones we split. The args are passed as-is.
        # If a model is to be made compatible with FasterCache, the user must ensure that the inputs that
        # contain batchwise-concatenated unconditional and conditional inputs are passed as kwargs.
        self.uncond_cond_input_kwargs_identifiers = uncond_cond_input_kwargs_identifiers
        self.tensor_format = tensor_format

    def _get_cond_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if should_skip_uncond:
            kwargs = {
                k: v if k not in self.uncond_cond_input_kwargs_identifiers else self._get_cond_input(v)
                for k, v in kwargs.items()
            }
            # TODO(aryan): remove later
            logger.debug("Skipping unconditional branch computation")

        if should_skip_uncond:
            breakpoint()
        output = module._old_forward(*args, **kwargs)
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

    def reset_state(self, module: nn.Module) -> None:
        module._fastercache_state.reset()


class FasterCacheBlockHook(ModelHook):
    _is_stateful = True

    def new_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)
        state: FasterCacheState = module._fastercache_state

        batch_size = [
            *[arg.size(0) for arg in args if torch.is_tensor(arg)],
            *[v.size(0) for v in kwargs.values() if torch.is_tensor(v)],
        ][0]
        if state.batch_size is None:
            # Will be updated on first forward pass through the denoiser
            state.batch_size = batch_size

        # If we have to skip due to the skip conditions, then let's skip as expected.
        # But, we can't skip if the denoiser wants to infer both unconditional and conditional branches. So,
        # if state.batch_size (which is the true unconditional-conditional batch size) is same as the current
        # batch size, we don't perform the layer skip. Otherwise, we conditionally skip the layer based on
        # what state.skip_callback returns.
        if state.skip_callback(module) and state.batch_size != batch_size:
            # TODO(aryan): remove later
            logger.debug("Skipping layer computation")
            t_2_output, t_output = state.cache

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

            output = t_output + (t_output - t_2_output) * state.weight_callback(module)
        else:
            output = module._old_forward(*args, **kwargs)

        # The output here can be both unconditional-conditional branch outputs or just conditional branch outputs.
        # This is determined at the higher-level denoiser module. We only want to cache the conditional branch outputs.
        cache_output = output
        if output.size(0) == state.batch_size:
            cache_output = cache_output.chunk(2, dim=0)[1]

        # Just to be safe that the output is of the correct size for both unconditional-conditional branch inference
        # and only-conditional branch inference.
        assert 2 * cache_output.size(0) == state.batch_size

        if state.cache is None:
            state.cache = [cache_output, cache_output]
        else:
            state.cache = [state.cache[-1], cache_output]

        state.iteration += 1
        return module._diffusers_hook.post_forward(module, output)

    def reset_state(self, module: nn.Module) -> None:
        module._fastercache_state.reset()


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
