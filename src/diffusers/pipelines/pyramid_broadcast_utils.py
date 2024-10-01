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

from typing import Tuple

import torch.nn as nn

from ..models.attention_processor import Attention, AttentionProcessor
from ..utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class PyramidAttentionBroadcastAttentionProcessor:
    def __init__(self, pipeline, processor: AttentionProcessor) -> None:
        self.pipeline = pipeline
        self._original_processor = processor
        self._prev_hidden_states = None
        self._iteration = 0

    def __call__(self, *args, **kwargs):
        if (
            hasattr(self.pipeline, "_current_timestep")
            and self.pipeline._current_timestep is not None
            and self._iteration % self.pipeline._pab_skip_range != 0
            and (
                self.pipeline._pab_timestep_range[0]
                < self.pipeline._current_timestep
                < self.pipeline._pab_timestep_range[1]
            )
        ):
            # print("Using cached states:", self.pipeline._current_timestep)
            hidden_states = self._prev_hidden_states
        else:
            hidden_states = self._original_processor(*args, **kwargs)
            self._prev_hidden_states = hidden_states

        self._iteration = (self._iteration + 1) % self.pipeline.num_timesteps

        return hidden_states


class PyramidAttentionBroadcastMixin:
    r"""Mixin class for [Pyramid Attention Broadcast](https://www.arxiv.org/abs/2408.12588)."""

    def _enable_pyramid_attention_broadcast(self) -> None:
        # def is_fake_integral_match(layer_id, name):
        #     layer_id = layer_id.split(".")[-1]
        #     name = name.split(".")[-1]
        #     return layer_id.isnumeric() and name.isnumeric() and layer_id == name

        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet

        for name, module in denoiser.named_modules():
            if isinstance(module, Attention):
                module.processor = PyramidAttentionBroadcastAttentionProcessor(self, module.processor)

        # target_modules = {}

        # for layer_id in self._pab_skip_range:
        #     for name, module in denoiser.named_modules():
        #         if (
        #             isinstance(module, Attention)
        #             and re.search(layer_id, name) is not None
        #             and not is_fake_integral_match(layer_id, name)
        #         ):
        #             target_modules[name] = module

        # for name, module in target_modules.items():
        #     # TODO: make this debug
        #     logger.info(f"Enabling Pyramid Attention Broadcast in layer: {name}")
        #     module.processor = PyramidAttentionBroadcastAttentionProcessor(self, module.processor)

    def _disable_pyramid_attention_broadcast(self) -> None:
        denoiser: nn.Module = self.transformer if hasattr(self, "transformer") else self.unet
        for name, module in denoiser.named_modules():
            if isinstance(module, Attention) and isinstance(
                module.processor, PyramidAttentionBroadcastAttentionProcessor
            ):
                # TODO: make this debug
                logger.info(f"Disabling Pyramid Attention Broadcast in layer: {name}")
                module.processor = module.processor._original_processor

    def enable_pyramid_attention_broadcast(self, skip_range: int, timestep_range: Tuple[int, int]) -> None:
        if isinstance(skip_range, str):
            skip_range = [skip_range]

        self._pab_skip_range = skip_range
        self._pab_timestep_range = timestep_range

        self._enable_pyramid_attention_broadcast()

    def disable_pyramid_attention_broadcast(self) -> None:
        self._pab_timestep_range = None
        self._pab_skip_range = None

    @property
    def pyramid_attention_broadcast_enabled(self):
        return hasattr(self, "_pab_skip_range") and self._pab_skip_range is not None
