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
from typing import Callable, List, Optional, Tuple

import torch.nn as nn

from ..models.attention_processor import Attention
from ..models.hooks import PyramidAttentionBroadcastHook, add_hook_to_module, remove_hook_from_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_ATTENTION_CLASSES = (Attention,)

_SPATIAL_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")
_TEMPORAL_ATTENTION_BLOCK_IDENTIFIERS = "temporal_transformer_blocks"
_CROSS_ATTENTION_BLOCK_IDENTIFIERS = ("blocks", "transformer_blocks")


@dataclass
class PyramidAttentionBroadcastConfig:
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
    def __init__(self) -> None:
        self.iteration = 0

    def reset_state(self):
        self.iteration = 0


def apply_pyramid_attention_broadcast(
    pipeline: DiffusionPipeline,
    config: Optional[PyramidAttentionBroadcastConfig] = None,
    denoiser: Optional[nn.Module] = None,
):
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
    block_skip_range: int,
    timestep_skip_range: Tuple[int, int],
    current_timestep_callback: Callable[[], int],
):
    module._pyramid_attention_broadcast_state = PyramidAttentionBroadcastState()
    min_timestep, max_timestep = timestep_skip_range

    def skip_callback(attention_module: nn.Module) -> bool:
        pab_state: PyramidAttentionBroadcastState = attention_module._pyramid_attention_broadcast_state
        current_timestep = current_timestep_callback()
        is_within_timestep_range = min_timestep < current_timestep < max_timestep

        if is_within_timestep_range:
            # As soon as the current timestep is within the timestep range, we start skipping attention computation.
            # The following inference steps will compute the attention every `block_skip_range` steps.
            should_compute_attention = pab_state.iteration > 0 and pab_state.iteration % block_skip_range == 0
            pab_state.iteration += 1
            print(current_timestep, is_within_timestep_range, should_compute_attention)
            return not should_compute_attention

        # We are still not yet in the phase of inference where skipping attention is possible without minimal quality
        # loss, as described in the paper. So, the attention computation cannot be skipped
        return False

    hook = PyramidAttentionBroadcastHook(skip_callback=skip_callback)
    add_hook_to_module(module, hook, append=True)


def _apply_pyramid_attention_broadcast_on_attention_class(
    pipeline: DiffusionPipeline, name: str, module: Attention, config: PyramidAttentionBroadcastConfig
):
    # Similar check as PEFT to determine if a string layer name matches a module name
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

    def current_timestep_callback():
        return pipeline._current_timestep

    apply_pyramid_attention_broadcast_on_module(
        module, block_skip_range, timestep_skip_range, current_timestep_callback
    )


class PyramidAttentionBroadcastMixin:
    r"""Mixin class for [Pyramid Attention Broadcast](https://www.arxiv.org/abs/2408.12588)."""

    def _enable_pyramid_attention_broadcast(self) -> None:
        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet

        for name, module in denoiser.named_modules():
            if isinstance(module, Attention):
                is_spatial_attention = (
                    any(x in name for x in self._pab_spatial_attn_layer_identifiers)
                    and self._pab_spatial_attn_skip_range is not None
                    and not module.is_cross_attention
                )
                is_temporal_attention = (
                    any(x in name for x in self._pab_temporal_attn_layer_identifiers)
                    and self._pab_temporal_attn_skip_range is not None
                    and not module.is_cross_attention
                )
                is_cross_attention = (
                    any(x in name for x in self._pab_cross_attn_layer_identifiers)
                    and self._pab_cross_attn_skip_range is not None
                    and module.is_cross_attention
                )

                if is_spatial_attention:
                    skip_range = self._pab_spatial_attn_skip_range
                    timestep_range = self._pab_spatial_attn_timestep_range
                if is_temporal_attention:
                    skip_range = self._pab_temporal_attn_skip_range
                    timestep_range = self._pab_temporal_attn_timestep_range
                if is_cross_attention:
                    skip_range = self._pab_cross_attn_skip_range
                    timestep_range = self._pab_cross_attn_timestep_range

                if skip_range is None:
                    continue

                # logger.debug(f"Enabling Pyramid Attention Broadcast in layer: {name}")
                print(f"Enabling Pyramid Attention Broadcast in layer: {name}")

                add_hook_to_module(
                    module,
                    PyramidAttentionBroadcastHook(
                        skip_range=skip_range,
                        timestep_range=timestep_range,
                        timestep_callback=self._pyramid_attention_broadcast_timestep_callback,
                    ),
                    append=True,
                )

    def _disable_pyramid_attention_broadcast(self) -> None:
        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet
        for name, module in denoiser.named_modules():
            logger.debug(f"Disabling Pyramid Attention Broadcast in layer: {name}")
            remove_hook_from_module(module)

    def _pyramid_attention_broadcast_timestep_callback(self):
        return self._current_timestep

    def enable_pyramid_attention_broadcast(
        self,
        spatial_attn_skip_range: Optional[int] = None,
        spatial_attn_timestep_range: Tuple[int, int] = (100, 800),
        temporal_attn_skip_range: Optional[int] = None,
        cross_attn_skip_range: Optional[int] = None,
        temporal_attn_timestep_range: Tuple[int, int] = (100, 800),
        cross_attn_timestep_range: Tuple[int, int] = (100, 800),
        spatial_attn_layer_identifiers: List[str] = ["blocks", "transformer_blocks"],
        temporal_attn_layer_identifiers: List[str] = ["temporal_transformer_blocks"],
        cross_attn_layer_identifiers: List[str] = ["blocks", "transformer_blocks"],
    ) -> None:
        r"""
        Enable pyramid attention broadcast to speedup inference by re-using attention states and skipping computation
        systematically as described in the paper: [Pyramid Attention Broadcast](https://www.arxiv.org/abs/2408.12588).

        Args:
            spatial_attn_skip_range (`int`, *optional*):
                The attention block to execute after skipping intermediate spatial attention blocks. If set to the
                value `N`, `N - 1` attention blocks are skipped and every N'th block is executed. Different models have
                different tolerances to how much attention computation can be reused based on the differences between
                successive blocks. So, this parameter must be adjusted per model after performing experimentation.
                Setting this value to `2` is recommended for different models PAB has been experimented with.
            temporal_attn_skip_range (`int`, *optional*):
                The attention block to execute after skipping intermediate temporal attention blocks. If set to the
                value `N`, `N - 1` attention blocks are skipped and every N'th block is executed. Different models have
                different tolerances to how much attention computation can be reused based on the differences between
                successive blocks. So, this parameter must be adjusted per model after performing experimentation.
                Setting this value to `4` is recommended for different models PAB has been experimented with.
            cross_attn_skip_range (`int`, *optional*):
                The attention block to execute after skipping intermediate cross attention blocks. If set to the value
                `N`, `N - 1` attention blocks are skipped and every N'th block is executed. Different models have
                different tolerances to how much attention computation can be reused based on the differences between
                successive blocks. So, this parameter must be adjusted per model after performing experimentation.
                Setting this value to `6` is recommended for different models PAB has been experimented with.
            spatial_attn_timestep_range (`Tuple[int, int]`, defaults to `(100, 800)`):
                The timestep range between which PAB will remain activated in spatial attention blocks. While
                activated, PAB will re-use attention computations between inference steps.
            temporal_attn_timestep_range (`Tuple[int, int]`, defaults to `(100, 800)`):
                The timestep range between which PAB will remain activated in temporal attention blocks. While
                activated, PAB will re-use attention computations between inference steps.
            cross_attn_timestep_range (`Tuple[int, int]`, defaults to `(100, 800)`):
                The timestep range between which PAB will remain activated in cross attention blocks. While activated,
                PAB will re-use attention computations between inference steps.
        """

        if spatial_attn_timestep_range[0] > spatial_attn_timestep_range[1]:
            raise ValueError(
                "Expected `spatial_attn_timestep_range` to be a tuple of two integers, with first value lesser or equal than second. These correspond to the min and max timestep between which PAB will be applied."
            )
        if temporal_attn_timestep_range[0] > temporal_attn_timestep_range[1]:
            raise ValueError(
                "Expected `temporal_attn_timestep_range` to be a tuple of two integers, with first value lesser or equal than second. These correspond to the min and max timestep between which PAB will be applied."
            )
        if cross_attn_timestep_range[0] > cross_attn_timestep_range[1]:
            raise ValueError(
                "Expected `cross_attn_timestep_range` to be a tuple of two integers, with first value lesser or equal than second. These correspond to the min and max timestep between which PAB will be applied."
            )

        self._pab_spatial_attn_skip_range = spatial_attn_skip_range
        self._pab_temporal_attn_skip_range = temporal_attn_skip_range
        self._pab_cross_attn_skip_range = cross_attn_skip_range
        self._pab_spatial_attn_timestep_range = spatial_attn_timestep_range
        self._pab_temporal_attn_timestep_range = temporal_attn_timestep_range
        self._pab_cross_attn_timestep_range = cross_attn_timestep_range
        self._pab_spatial_attn_layer_identifiers = spatial_attn_layer_identifiers
        self._pab_temporal_attn_layer_identifiers = temporal_attn_layer_identifiers
        self._pab_cross_attn_layer_identifiers = cross_attn_layer_identifiers

        self._pab_enabled = spatial_attn_skip_range or temporal_attn_skip_range or cross_attn_skip_range

        self._enable_pyramid_attention_broadcast()

    def disable_pyramid_attention_broadcast(self) -> None:
        r"""Disables the pyramid attention broadcast sampling mechanism."""

        self._pab_spatial_attn_skip_range = None
        self._pab_temporal_attn_skip_range = None
        self._pab_cross_attn_skip_range = None
        self._pab_spatial_attn_timestep_range = None
        self._pab_temporal_attn_timestep_range = None
        self._pab_cross_attn_timestep_range = None
        self._pab_spatial_attn_layer_identifiers = None
        self._pab_temporal_attn_layer_identifiers = None
        self._pab_cross_attn_layer_identifiers = None
        self._pab_enabled = False

    @property
    def pyramid_attention_broadcast_enabled(self):
        return hasattr(self, "_pab_enabled") and self._pab_enabled
