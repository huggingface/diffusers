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

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.fft as FFT
import torch.nn as nn

from ..models.attention_processor import Attention
from ..models.hooks import ModelHook, add_hook_to_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_ATTENTION_CLASSES = (Attention,)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks",)


@dataclass
class FasterCacheConfig:
    r"""
    Configuration for [FasterCache](https://huggingface.co/papers/2410.19355).
    """

    # In the paper and codebase, they hardcode these values to 2. However, it can be made configurable
    # after some testing. We default to 2 if these parameters are not provided.
    spatial_attention_block_skip_range: Optional[int] = None
    temporal_attention_block_skip_range: Optional[int] = None

    # TODO(aryan): write heuristics for what the best way to obtain these values are
    spatial_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (-1, 681)

    # Indicator functions for low/high frequency as mentioned in Equation 11 of the paper
    low_frequency_weight_update_timestep_range: Tuple[int, int] = (99, 641)
    high_frequency_weight_update_timestep_range: Tuple[int, int] = (-1, 301)

    # ⍺1 and ⍺2 as mentioned in Equation 11 of the paper
    alpha_low_frequency = 1.1
    alpha_high_frequency = 1.1

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS

    attention_weight_callback: Callable[[nn.Module], float] = None
    low_frequency_weight_callback: Callable[[nn.Module], float] = None
    high_frequency_weight_callback: Callable[[nn.Module], float] = None


class FasterCacheDenoiserState:
    r"""
    State for [FasterCache](https://huggingface.co/papers/2410.19355) top-level denoiser module.
    """

    def __init__(self, delta_update_callback: Callable[[Any, int, float, float], Tuple[float, float]]) -> None:
        self.delta_update_callback = delta_update_callback

        self.iteration = 0
        self.low_frequency_delta = None
        self.high_frequency_delta = None

    def update_state(self, output: Any) -> None:
        self.iteration += 1
        self.low_frequency_delta, self.high_frequency_delta = self.delta_update_callback(
            output, self.iteration, self.low_frequency_delta, self.high_frequency_delta
        )

    def reset_state(self):
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

    def __init__(self) -> None:
        self.iteration = 0
        self.cache = None

    def update_state(self, output: Any) -> None:
        self.iteration += 1
        if self.cache is None:
            self.cache = [output, output]
        else:
            self.cache = [self.cache[-1], output]

    def reset_state(self):
        self.iteration = 0
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
            "FasterCache requires one of `spatial_attention_block_skip_range` or `temporal_attention_block_skip_range` "
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
        config.low_frequency_weight_callback = lambda _: config.alpha_low_frequency

    if config.high_frequency_weight_callback is None:
        logger.debug(
            "High frequency weight callback not provided when enabling FasterCache. Defaulting to behaviour described in the paper."
        )
        config.high_frequency_weight_callback = lambda _: config.alpha_high_frequency

    if denoiser is None:
        denoiser = pipeline.transformer if hasattr(pipeline, "transformer") else pipeline.unet

    for name, module in denoiser.named_modules():
        if not isinstance(module, _ATTENTION_CLASSES):
            continue
        if isinstance(module, Attention):
            _apply_fastercache_on_attention_class(pipeline, name, module, config)


def apply_fastercache_on_module(
    module: nn.Module, skip_callback: Callable[[nn.Module], bool], weight_callback: Callable[[nn.Module], float]
) -> None:
    module._fastercache_state = FasterCacheState()
    hook = FasterCacheBlockHook(skip_callback, weight_callback)
    add_hook_to_module(module, hook, append=True)


def _apply_fastercache_on_attention_class(
    pipeline: DiffusionPipeline, name: str, module: Attention, config: FasterCacheConfig
) -> None:
    # Similar check as PEFT to determine if a string layer name matches a module name
    # TODO(aryan): make this regex based
    is_spatial_self_attention = (
        any(
            f"{identifier}." in name or identifier == name for identifier in config.spatial_attention_block_identifiers
        )
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
        logger.info(
            f'Unable to apply FasterCache to the selected layer: "{name}" because it does '
            f"not match any of the required criteria for spatial, temporal or cross attention layers. Note, "
            f"however, that this layer may still be valid for applying PAB. Please specify the correct "
            f"block identifiers in the configuration or use the specialized `apply_fastercache_on_module` "
            f"function to apply FasterCache to this layer."
        )
        return

    def skip_callback(module: nn.Module) -> bool:
        is_using_classifier_free_guidance = pipeline.do_classifier_free_guidance
        if not is_using_classifier_free_guidance:
            return False

        fastercache_state = module._fastercache_state
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
    apply_fastercache_on_module(module, skip_callback, config.attention_weight_callback)


class FasterCacheModelHook(ModelHook):
    def __init__(self) -> None:
        super().__init__()


class FasterCacheBlockHook(ModelHook):
    def __init__(
        self, skip_callback: Callable[[nn.Module], bool], weight_callback: Callable[[nn.Module], float]
    ) -> None:
        super().__init__()

        self.skip_callback = skip_callback
        self.weight_callback = weight_callback

    def new_forward(self, module: nn.Module, *args, **kwargs) -> Any:
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)

        if self.skip_callback(module):
            t_2_output, t_output = module._fastercache_state.cache
            output = t_output + (t_output - t_2_output) * self.weight_callback(module)
        else:
            output = module._old_forward(*args, **kwargs)

        return module._diffusers_hook.post_forward(module, output)

    def post_forward(self, module: nn.Module, output: Any) -> Any:
        module._fastercache_state.update_state(output)
        return output


# Reference: https://github.com/Vchitect/FasterCache/blob/fab32c15014636dc854948319c0a9a8d92c7acb4/scripts/latte/fastercache_sample_latte.py#L127C1-L143C39
@torch.no_grad()
def _fft(tensor):
    tensor_fft = FFT.fft2(tensor)
    tensor_fft_shifted = FFT.fftshift(tensor_fft)
    batch_size, num_channels, height, width = tensor.size()
    radius = min(height, width) // 5

    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    center_x, center_y = width // 2, height // 2
    mask = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2 <= radius**2

    low_freq_mask = mask.unsqueeze(0).unsqueeze(0).to(tensor.device)
    high_freq_mask = ~low_freq_mask

    low_freq_fft = tensor_fft_shifted * low_freq_mask
    high_freq_fft = tensor_fft_shifted * high_freq_mask

    return low_freq_fft, high_freq_fft
