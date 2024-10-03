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
        _supported_parameters = {"attn", "hidden_states", "encoder_hidden_states", "attention_mask", "temb", "image_rotary_emb"}
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
        if (
            hasattr(self.pipeline, "_current_timestep")
            and self.pipeline._current_timestep is not None
            and self._iteration % self._skip_range != 0
            and (self._timestep_range[0] < self.pipeline._current_timestep < self._timestep_range[1])
        ):
            hidden_states = self._prev_hidden_states
        else:
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

                module.set_processor(PyramidAttentionBroadcastAttentionProcessorWrapper(
                    self, module.processor, skip_range, timestep_range
                ))

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
        self._pab_spatial_attn_skip_range = None
        self._pab_cross_attn_skip_range = None
        self._pab_spatial_attn_timestep_range = None
        self._pab_cross_attn_timestep_range = None
        self._pab_enabled = False

    @property
    def pyramid_attention_broadcast_enabled(self):
        return hasattr(self, "_pab_enabled") and self._pab_enabled
