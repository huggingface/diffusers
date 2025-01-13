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
from typing import Any, Callable, Optional, Tuple, Union

import torch

from ..models.attention_processor import Attention, MochiAttention
from ..utils import logging
from .hooks import HookRegistry, ModelHook


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_ATTENTION_CLASSES = (Attention, MochiAttention)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks", "single_transformer_blocks")
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = ("temporal_transformer_blocks",)
_CROSS_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")


@dataclass
class PyramidAttentionBroadcastConfig:
    r"""
    Configuration for Pyramid Attention Broadcast.

    Args:
        spatial_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of times a specific spatial attention broadcast is skipped before computing the attention states
            to re-use. If this is set to the value `N`, the attention computation will be skipped `N - 1` times (i.e.,
            old attention states will be re-used) before computing the new attention states again.
        temporal_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of times a specific temporal attention broadcast is skipped before computing the attention
            states to re-use. If this is set to the value `N`, the attention computation will be skipped `N - 1` times
            (i.e., old attention states will be re-used) before computing the new attention states again.
        cross_attention_block_skip_range (`int`, *optional*, defaults to `None`):
            The number of times a specific cross-attention broadcast is skipped before computing the attention states
            to re-use. If this is set to the value `N`, the attention computation will be skipped `N - 1` times (i.e.,
            old attention states will be re-used) before computing the new attention states again.
        spatial_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the spatial attention layer. The attention computations will be
            conditionally skipped if the current timestep is within the specified range.
        temporal_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the temporal attention layer. The attention computations will be
            conditionally skipped if the current timestep is within the specified range.
        cross_attention_timestep_skip_range (`Tuple[int, int]`, defaults to `(100, 800)`):
            The range of timesteps to skip in the cross-attention layer. The attention computations will be
            conditionally skipped if the current timestep is within the specified range.
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

    current_timestep_callback: Callable[[], int] = None

    # TODO(aryan): add PAB for MLP layers (very limited speedup from testing with original codebase
    # so not added for now)


class PyramidAttentionBroadcastState:
    r"""
    State for Pyramid Attention Broadcast.

    Attributes:
        iteration (`int`):
            The current iteration of the Pyramid Attention Broadcast. It is necessary to ensure that `reset_state` is
            called before starting a new inference forward pass for PAB to work correctly.
        cache (`Any`):
            The cached output from the previous forward pass. This is used to re-use the attention states when the
            attention computation is skipped. It is either a tensor or a tuple of tensors, depending on the module.
    """

    def __init__(self) -> None:
        self.iteration = 0
        self.cache = None

    def reset(self):
        self.iteration = 0
        self.cache = None

    def __repr__(self):
        cache_repr = ""
        if self.cache is None:
            cache_repr = "None"
        else:
            cache_repr = f"Tensor(shape={self.cache.shape}, dtype={self.cache.dtype})"
        return f"PyramidAttentionBroadcastState(iteration={self.iteration}, cache={cache_repr})"


class PyramidAttentionBroadcastHook(ModelHook):
    r"""A hook that applies Pyramid Attention Broadcast to a given module."""

    _is_stateful = True

    def __init__(self, skip_callback: Callable[[torch.nn.Module], bool]) -> None:
        super().__init__()

        self.skip_callback = skip_callback

    def initialize_hook(self, module):
        self.state = PyramidAttentionBroadcastState()
        return module

    def new_forward(self, module: torch.nn.Module, *args, **kwargs) -> Any:
        if self.skip_callback(module):
            output = self.state.cache
        else:
            output = module._old_forward(*args, **kwargs)

        self.state.cache = output
        self.state.iteration += 1
        return output

    def reset_state(self, module: torch.nn.Module) -> None:
        self.state.reset()
        return module


def apply_pyramid_attention_broadcast(
    module: torch.nn.Module,
    config: PyramidAttentionBroadcastConfig,
):
    r"""
    Apply [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588) to a given pipeline.

    PAB is an attention approximation method that leverages the similarity in attention states between timesteps to
    reduce the computational cost of attention computation. The key takeaway from the paper is that the attention
    similarity in the cross-attention layers between timesteps is high, followed by less similarity in the temporal and
    spatial layers. This allows for the skipping of attention computation in the cross-attention layers more frequently
    than in the temporal and spatial layers. Applying PAB will, therefore, speedup the inference process.

    Args:
        module (`torch.nn.Module`):
            The module to apply Pyramid Attention Broadcast to.
        config (`Optional[PyramidAttentionBroadcastConfig]`, `optional`, defaults to `None`):
            The configuration to use for Pyramid Attention Broadcast.

    Example:

    ```python
    >>> import torch
    >>> from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
    >>> from diffusers.utils import export_to_video

    >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
    >>> pipe.to("cuda")

    >>> config = PyramidAttentionBroadcastConfig(
    ...     spatial_attention_block_skip_range=2,
    ...     spatial_attention_timestep_skip_range=(100, 800),
    ...     current_timestep_callback=lambda: pipe._current_timestep,
    ... )
    >>> apply_pyramid_attention_broadcast(pipe.transformer, config)
    ```
    """
    if config.current_timestep_callback is None:
        raise ValueError(
            "The `current_timestep_callback` function must be provided in the configuration to apply Pyramid Attention Broadcast."
        )

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

    for name, submodule in module.named_modules():
        if not isinstance(submodule, _ATTENTION_CLASSES):
            continue
        if isinstance(submodule, Attention):
            _apply_pyramid_attention_broadcast_on_attention_class(name, submodule, config)
        if isinstance(submodule, MochiAttention):
            _apply_pyramid_attention_broadcast_on_mochi_attention_class(name, submodule, config)


def _apply_pyramid_attention_broadcast_on_attention_class(
    name: str, module: Attention, config: PyramidAttentionBroadcastConfig
) -> bool:
    is_spatial_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.spatial_attention_block_identifiers)
        and config.spatial_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_temporal_self_attention = (
        any(re.search(identifier, name) is not None for identifier in config.temporal_attention_block_identifiers)
        and config.temporal_attention_block_skip_range is not None
        and not getattr(module, "is_cross_attention", False)
    )
    is_cross_attention = (
        any(re.search(identifier, name) is not None for identifier in config.cross_attention_block_identifiers)
        and config.cross_attention_block_skip_range is not None
        and getattr(module, "is_cross_attention", False)
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
    elif is_cross_attention:
        block_skip_range = config.cross_attention_block_skip_range
        timestep_skip_range = config.cross_attention_timestep_skip_range
        block_type = "cross"

    if block_skip_range is None or timestep_skip_range is None:
        logger.info(
            f'Unable to apply Pyramid Attention Broadcast to the selected layer: "{name}" because it does '
            f"not match any of the required criteria for spatial, temporal or cross attention layers. Note, "
            f"however, that this layer may still be valid for applying PAB. Please specify the correct "
            f"block identifiers in the configuration or use the specialized `apply_pyramid_attention_broadcast_on_module` "
            f"function to apply PAB to this layer."
        )
        return False

    def skip_callback(module: torch.nn.Module) -> bool:
        hook: PyramidAttentionBroadcastHook = module._diffusers_hook.get_hook("pyramid_attention_broadcast")
        pab_state: PyramidAttentionBroadcastState = hook.state

        if pab_state.cache is None:
            return False

        is_within_timestep_range = timestep_skip_range[0] < config.current_timestep_callback() < timestep_skip_range[1]
        if not is_within_timestep_range:
            # We are still not in the phase of inference where skipping attention is possible without minimal quality
            # loss, as described in the paper. So, the attention computation cannot be skipped
            return False

        should_compute_attention = pab_state.iteration > 0 and pab_state.iteration % block_skip_range == 0
        return not should_compute_attention

    logger.debug(f"Enabling Pyramid Attention Broadcast ({block_type}) in layer: {name}")
    _apply_pyramid_attention_broadcast(module, skip_callback)
    return True


def _apply_pyramid_attention_broadcast_on_mochi_attention_class(
    name: str, module: MochiAttention, config: PyramidAttentionBroadcastConfig
) -> bool:
    # The same logic as Attention class works here, so just use that for now
    return _apply_pyramid_attention_broadcast_on_attention_class(name, module, config)


def _apply_pyramid_attention_broadcast(
    module: Union[Attention, MochiAttention],
    skip_callback: Callable[[torch.nn.Module], bool],
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
    registry = HookRegistry.check_if_exists_or_initialize(module)
    hook = PyramidAttentionBroadcastHook(skip_callback)
    registry.register_hook(hook, "pyramid_attention_broadcast")
