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
from typing import Callable, Optional, Protocol, Tuple

import torch.nn as nn

from ..models.attention_processor import Attention
from ..models.hooks import PyramidAttentionBroadcastHook, add_hook_to_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_ATTENTION_CLASSES = (Attention,)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = "temporal_transformer_blocks"
_CROSS_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")


@dataclass
class PyramidAttentionBroadcastConfig:
    r"""
    Configuration for Pyramid Attention Broadcast.

    Args:
        spatial_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of blocks to skip in the spatial attention layer. If `None`, the spatial attention layer
            computations will not be skipped.
        temporal_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of blocks to skip in the temporal attention layer. If `None`, the temporal attention layer
            computations will not be skipped.
        cross_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of blocks to skip in the cross-attention layer. If `None`, the cross-attention layer computations
            will not be skipped.
        spatial_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the spatial attention layer. The attention computations will be skipped
            if the current timestep is within the specified range.
        temporal_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the temporal attention layer. The attention computations will be skipped
            if the current timestep is within the specified range.
        cross_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the cross-attention layer. The attention computations will be skipped if
            the current timestep is within the specified range.
        spatial_attention_block_identifiers (`Tuple[str, ...]`, defaults to `("blocks", "transformer_blocks")`):
            The identifiers to match against the layer names to determine if the layer is a spatial attention layer.
        temporal_attention_block_identifiers (`Tuple[str, ...]`, defaults to `("temporal_transformer_blocks",)`):
            The identifiers to match against the layer names to determine if the layer is a temporal attention layer.
        cross_attention_block_identifiers (`Tuple[str, ...]`, defaults to `("blocks", "transformer_blocks")`):
            The identifiers to match against the layer names to determine if the layer is a cross-attention layer.
    """
    spatial_attention_block_skip_range: Optional[int] = None
    temporal_attention_block_skip_range: Optional[int] = None
    cross_attention_block_skip_range: Optional[int] = None

    spatial_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    temporal_attention_timestep_skip_range: Tuple[int, int] = (100, 800)
    cross_attention_timestep_skip_range: Tuple[int, int] = (100, 800)

    spatial_attention_block_identifiers: Tuple[str, ...] = _SPATIAL_ATTENTION_BLOCK_IDENTIFIERS
    temporal_attention_block_identifiers: Tuple[str, ...] = _TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS
    cross_attention_block_identifiers: Tuple[str, ...] = _CROSS_ATTENTION_BLOCK_IDENTIFIERS


class PyramidAttentionBroadcastState:
    r"""
    State for Pyramid Attention Broadcast.

    Attributes:
        iteration (`int`):
            The current iteration of the Pyramid Attention Broadcast. It is necessary to ensure that `reset_state` is
            called before starting a new inference forward pass for PAB to work correctly.
    """
    def __init__(self) -> None:
        self.iteration = 0

    def reset_state(self):
        self.iteration = 0


class nnModulePAB(Protocol):
    r"""
    Type hint for a torch.nn.Module that contains a `_pyramid_attention_broadcast_state` attribute.

    Attributes:
        _pyramid_attention_broadcast_state (`PyramidAttentionBroadcastState`):
            The state of Pyramid Attention Broadcast.
    """
    _pyramid_attention_broadcast_state: PyramidAttentionBroadcastState


def apply_pyramid_attention_broadcast(
    pipeline: DiffusionPipeline,
    config: Optional[PyramidAttentionBroadcastConfig] = None,
    denoiser: Optional[nn.Module] = None,
):
    r"""
    Apply [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588) to a given pipeline.

    PAB is an attention approximation method that leverages the similarity in attention states between timesteps to
    reduce the computational cost of attention computation. The key takeaway from the paper is that the attention
    similarity in the cross-attention layers between timesteps is high, followed by less similarity in the temporal and
    spatial layers. This allows for the skipping of attention computation in the cross-attention layers more frequently
    than in the temporal and spatial layers. Applying PAB will, therefore, speedup the inference process.

    Args:
        pipeline (`DiffusionPipeline`):
            The diffusion pipeline to apply Pyramid Attention Broadcast to.
        config (`Optional[PyramidAttentionBroadcastConfig]`, `optional`, defaults to `None`):
            The configuration to use for Pyramid Attention Broadcast.
        denoiser (`Optional[nn.Module]`, `optional`, defaults to `None`):
            The denoiser module to apply Pyramid Attention Broadcast to. If `None`, the pipeline's transformer or unet
            module will be used.

    Example:

    ```python
    >>> from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast

    >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> config = PyramidAttentionBroadcastConfig(
    ...     spatial_attention_block_skip_range=2, spatial_attention_timestep_skip_range=(100, 800)
    ... )
    >>> apply_pyramid_attention_broadcast(pipe, config)
    ```
    """
    # We present Pyramid Attention Broadcast (PAB), a real-time, high quality and training-free approach for DiT-based video generation. Our method is founded on the observation that attention difference in the diffusion process exhibits a U-shaped pattern, indicating significant redundancy. We mitigate this by broadcasting attention outputs to subsequent steps in a pyramid style. It applies different broadcast strategies to each attention based on their variance for best efficiency. We further introduce broadcast sequence parallel for more efficient distributed inference. PAB demonstrates superior results across three models compared to baselines, achieving real-time generation for up to 720p videos. We anticipate that our simple yet effective method will serve as a robust baseline and facilitate future research and application for video generation.
    if config is None:
        config = PyramidAttentionBroadcastConfig()

    if (
        config.spatial_attention_block_skip_range is None
        and config.temporal_attention_block_skip_range is None
        and config.cross_attention_block_skip_range is None
    ):
        logger.warning(
            "Pyramid Attention Broadcast requires one or more of `spatial_attention_block_skip_range`, `temporal_attention_block_skip_range` "
            "or `cross_attention_block_skip_range` parameters to be set to an integer, not `None`. Defaulting to using `spatial_attention_block_skip_range=2`. "
            "To avoid this warning, please set one of the above parameters."
        )
        config.spatial_attention_block_skip_range = 2

    if denoiser is None:
        denoiser = pipeline.transformer if hasattr(pipeline, "transformer") else pipeline.unet

    for name, module in denoiser.named_modules():
        if not isinstance(module, _ATTENTION_CLASSES):
            continue
        if isinstance(module, Attention):
            _apply_pyramid_attention_broadcast_on_attention_class(pipeline, name, module, config)


def apply_pyramid_attention_broadcast_on_module(
    module: Attention,
    skip_callback: Callable[[nn.Module], bool],
):
    r"""
    Apply [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588) to a given torch.nn.Module.

    Args:
        module (`torch.nn.Module`):
            The module to apply Pyramid Attention Broadcast to.
        skip_callback (`Callable[[nn.Module], bool]`):
            A callback function that determines whether the attention computation should be skipped or not. The
            callback function should return a boolean value, where `True` indicates that the attention computation
            should be skipped, and `False` indicates that the attention computation should not be skipped. The callback
            function will receive a torch.nn.Module containing a `_pyramid_attention_broadcast_state` attribute that
            can should be used to retrieve and update the state of PAB for the given module.
    """
    module._pyramid_attention_broadcast_state = PyramidAttentionBroadcastState()
    hook = PyramidAttentionBroadcastHook(skip_callback=skip_callback)
    add_hook_to_module(module, hook, append=True)


def _apply_pyramid_attention_broadcast_on_attention_class(
    pipeline: DiffusionPipeline, name: str, module: Attention, config: PyramidAttentionBroadcastConfig
):
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
    is_cross_attention = (
        any(f"{identifier}." in name or identifier == name for identifier in config.cross_attention_block_identifiers)
        and config.cross_attention_block_skip_range is not None
        and not module.is_cross_attention
    )

    block_skip_range, timestep_skip_range = None, None
    if is_spatial_self_attention:
        block_skip_range = config.spatial_attention_block_skip_range
        timestep_skip_range = config.spatial_attention_timestep_skip_range
    elif is_temporal_self_attention:
        block_skip_range = config.temporal_attention_block_skip_range
        timestep_skip_range = config.temporal_attention_timestep_skip_range
    elif is_cross_attention:
        block_skip_range = config.cross_attention_block_skip_range
        timestep_skip_range = config.cross_attention_timestep_skip_range

    if block_skip_range is None or timestep_skip_range is None:
        logger.warning(f"Unable to apply Pyramid Attention Broadcast to the selected layer: {name}.")
        return

    def skip_callback(module: nnModulePAB) -> bool:
        pab_state = module._pyramid_attention_broadcast_state
        current_timestep = pipeline._current_timestep
        is_within_timestep_range = timestep_skip_range[0] < current_timestep < timestep_skip_range[1]

        if is_within_timestep_range:
            should_compute_attention = pab_state.iteration > 0 and pab_state.iteration % block_skip_range == 0
            pab_state.iteration += 1
            return not should_compute_attention

        # We are still not yet in the phase of inference where skipping attention is possible without minimal quality
        # loss, as described in the paper. So, the attention computation cannot be skipped
        return False

    logger.debug(f"Enabling Pyramid Attention Broadcast in layer: {name}")
    apply_pyramid_attention_broadcast_on_module(module, skip_callback)
