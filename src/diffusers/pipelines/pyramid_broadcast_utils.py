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

import inspect
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..models.attention_processor import Attention, AttentionProcessor
from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PyramidAttentionBroadcastAttentionProcessorWrapper:
    r"""
    Helper attention processor that wraps logic required for Pyramid Attention Broadcast to function.

    PAB works by caching and re-using attention computations from past inference steps. This is due to the realization
    that the attention states do not differ too much numerically between successive inference steps. The difference is
    most significant/prominent in the spatial attention blocks, lesser so in the temporal attention blocks, and least
    in cross attention blocks.

    Currently, only spatial and cross attention block skipping is supported in Diffusers due to not having any models
    tested with temporal attention blocks. Feel free to open a PR adding support for this in case there's a model that
    you would like to use PAB with.

    Args:
        pipeline ([`~diffusers.DiffusionPipeline`]):
            The underlying DiffusionPipeline object that inherits from the PAB Mixin and utilized this attention
            processor.
        processor ([`~diffusers.models.attention_processor.AttentionProcessor`]):
            The underlying attention processor that will be wrapped to cache the intermediate attention computation.
        skip_range (`int`):
            The attention block to execute after skipping intermediate attention blocks. If set to the value `N`, `N -
            1` attention blocks are skipped and every N'th block is executed. Different models have different
            tolerances to how much attention computation can be reused based on the differences between successive
            blocks. So, this parameter must be adjusted per model after performing experimentation. Setting this value
            to `2` is recommended for different models PAB has been experimented with.
        timestep_range (`Tuple[int, int]`):
            The timestep range between which PAB will remain activated in attention blocks. While activated, PAB will
            re-use attention computations between inference steps.
    """

    def __init__(
        self, pipeline, processor: AttentionProcessor, skip_range: int, timestep_range: Tuple[int, int]
    ) -> None:
        self.pipeline = pipeline
        self._original_processor = processor
        self._skip_range = skip_range
        self._timestep_range = timestep_range

        self._prev_hidden_states = None
        self._iteration = 0

        _original_processor_params = set(inspect.signature(self._original_processor).parameters.keys())
        _supported_parameters = {
            "attn",
            "hidden_states",
            "encoder_hidden_states",
            "attention_mask",
            "temb",
            "image_rotary_emb",
        }
        self._attn_processor_params = _supported_parameters.intersection(_original_processor_params)

    def __call__(
        self,
        attn: Attention,
        hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        r"""Method that wraps the underlying call to compute attention and cache states for re-use."""

        if (
            hasattr(self.pipeline, "_current_timestep")
            and self.pipeline._current_timestep is not None
            and self._iteration % self._skip_range != 0
            and (self._timestep_range[0] < self.pipeline._current_timestep < self._timestep_range[1])
        ):
            # Skip attention computation by re-using past attention states
            hidden_states = self._prev_hidden_states
        else:
            # Perform attention computation
            call_kwargs = {}
            for param in self._attn_processor_params:
                call_kwargs.update({param: locals()[param]})
            call_kwargs.update(kwargs)
            hidden_states = self._original_processor(*args, **call_kwargs)
            self._prev_hidden_states = hidden_states

        self._iteration = (self._iteration + 1) % self.pipeline.num_timesteps

        return hidden_states


class PyramidAttentionBroadcastMixin:
    r"""Mixin class for [Pyramid Attention Broadcast](https://www.arxiv.org/abs/2408.12588)."""

    def _enable_pyramid_attention_broadcast(self) -> None:
        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet

        for name, module in denoiser.named_modules():
            if isinstance(module, Attention):
                logger.debug(f"Enabling Pyramid Attention Broadcast in layer: {name}")

                skip_range, timestep_range = None, None
                if module.is_cross_attention and self._pab_cross_attn_skip_range is not None:
                    skip_range = self._pab_cross_attn_skip_range
                    timestep_range = self._pab_cross_attn_timestep_range
                if not module.is_cross_attention and self._pab_spatial_attn_skip_range is not None:
                    skip_range = self._pab_spatial_attn_skip_range
                    timestep_range = self._pab_spatial_attn_timestep_range

                if skip_range is None:
                    continue

                module.set_processor(
                    PyramidAttentionBroadcastAttentionProcessorWrapper(
                        self, module.processor, skip_range, timestep_range
                    )
                )

    def _disable_pyramid_attention_broadcast(self) -> None:
        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet
        for name, module in denoiser.named_modules():
            if isinstance(module, Attention) and isinstance(
                module.processor, PyramidAttentionBroadcastAttentionProcessorWrapper
            ):
                logger.debug(f"Disabling Pyramid Attention Broadcast in layer: {name}")
                module.processor = module.processor._original_processor

    def enable_pyramid_attention_broadcast(
        self,
        spatial_attn_skip_range: Optional[int] = None,
        cross_attn_skip_range: Optional[int] = None,
        spatial_attn_timestep_range: Optional[Tuple[int, int]] = None,
        cross_attn_timestep_range: Optional[Tuple[int, int]] = None,
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
            cross_attn_skip_range (`int`, *optional*):
                The attention block to execute after skipping intermediate cross attention blocks. If set to the value
                `N`, `N - 1` attention blocks are skipped and every N'th block is executed. Different models have
                different tolerances to how much attention computation can be reused based on the differences between
                successive blocks. So, this parameter must be adjusted per model after performing experimentation.
                Setting this value to `6` is recommended for different models PAB has been experimented with.
            spatial_attn_timestep_range (`Tuple[int, int]`, *optional*):
                The timestep range between which PAB will remain activated in spatial attention blocks. While
                activated, PAB will re-use attention computations between inference steps. Setting this to `(100, 850)`
                is recommended for different models PAB has been experimented with.
            cross_attn_timestep_range (`Tuple[int, int]`, *optional*):
                The timestep range between which PAB will remain activated in cross attention blocks. While activated,
                PAB will re-use attention computations between inference steps. Setting this to `(100, 800)` is
                recommended for different models PAB has been experimented with.
        """

        if spatial_attn_timestep_range is None:
            spatial_attn_timestep_range = (100, 800)
        if cross_attn_skip_range is None:
            cross_attn_timestep_range = (100, 800)

        if spatial_attn_timestep_range[0] > spatial_attn_timestep_range[1]:
            raise ValueError(
                "Expected `spatial_attn_timestep_range` to be a tuple of two integers, with first value lesser or equal than second. These correspond to the min and max timestep between which PAB will be applied."
            )
        if cross_attn_timestep_range[0] > cross_attn_timestep_range[1]:
            raise ValueError(
                "Expected `cross_attn_timestep_range` to be a tuple of two integers, with first value lesser or equal than second. These correspond to the min and max timestep between which PAB will be applied."
            )

        self._pab_spatial_attn_skip_range = spatial_attn_skip_range
        self._pab_cross_attn_skip_range = cross_attn_skip_range
        self._pab_spatial_attn_timestep_range = spatial_attn_timestep_range
        self._pab_cross_attn_timestep_range = cross_attn_timestep_range
        self._pab_enabled = spatial_attn_skip_range or cross_attn_skip_range

        self._enable_pyramid_attention_broadcast()

    def disable_pyramid_attention_broadcast(self) -> None:
        r"""Disables the pyramid attention broadcast sampling mechanism."""

        self._pab_spatial_attn_skip_range = None
        self._pab_cross_attn_skip_range = None
        self._pab_spatial_attn_timestep_range = None
        self._pab_cross_attn_timestep_range = None
        self._pab_enabled = False

    @property
    def pyramid_attention_broadcast_enabled(self):
        return hasattr(self, "_pab_enabled") and self._pab_enabled
