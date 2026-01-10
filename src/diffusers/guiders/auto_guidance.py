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

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from ..hooks import HookRegistry, LayerSkipConfig
from ..hooks.layer_skip import _apply_layer_skip_hook
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class AutoGuidance(BaseGuidance):
    """
    AutoGuidance: https://huggingface.co/papers/2406.02507

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        auto_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply skip layer guidance to. Can be a single integer or a list of integers. If not
            provided, `skip_layer_config` must be provided.
        auto_guidance_config (`LayerSkipConfig` or `List[LayerSkipConfig]`, *optional*):
            The configuration for the skip layer guidance. Can be a single `LayerSkipConfig` or a list of
            `LayerSkipConfig`. If not provided, `skip_layer_guidance_layers` must be provided.
        dropout (`float`, *optional*):
            The dropout probability for autoguidance on the enabled skip layers (either with `auto_guidance_layers` or
            `auto_guidance_config`). If not provided, the dropout probability will be set to 1.0.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.0`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `1.0`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    _input_predictions = ["pred_cond", "pred_uncond"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        auto_guidance_layers: Optional[Union[int, List[int]]] = None,
        auto_guidance_config: Union[LayerSkipConfig, List[LayerSkipConfig], Dict[str, Any]] = None,
        dropout: Optional[float] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.auto_guidance_layers = auto_guidance_layers
        self.auto_guidance_config = auto_guidance_config
        self.dropout = dropout
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        is_layer_or_config_provided = auto_guidance_layers is not None or auto_guidance_config is not None
        is_layer_and_config_provided = auto_guidance_layers is not None and auto_guidance_config is not None
        if not is_layer_or_config_provided:
            raise ValueError(
                "Either `auto_guidance_layers` or `auto_guidance_config` must be provided to enable AutoGuidance."
            )
        if is_layer_and_config_provided:
            raise ValueError("Only one of `auto_guidance_layers` or `auto_guidance_config` can be provided.")
        if auto_guidance_config is None and dropout is None:
            raise ValueError("`dropout` must be provided if `auto_guidance_layers` is provided.")

        if auto_guidance_layers is not None:
            if isinstance(auto_guidance_layers, int):
                auto_guidance_layers = [auto_guidance_layers]
            if not isinstance(auto_guidance_layers, list):
                raise ValueError(
                    f"Expected `auto_guidance_layers` to be an int or a list of ints, but got {type(auto_guidance_layers)}."
                )
            auto_guidance_config = [
                LayerSkipConfig(layer, fqn="auto", dropout=dropout) for layer in auto_guidance_layers
            ]

        if isinstance(auto_guidance_config, dict):
            auto_guidance_config = LayerSkipConfig.from_dict(auto_guidance_config)

        if isinstance(auto_guidance_config, LayerSkipConfig):
            auto_guidance_config = [auto_guidance_config]

        if not isinstance(auto_guidance_config, list):
            raise ValueError(
                f"Expected `auto_guidance_config` to be a LayerSkipConfig or a list of LayerSkipConfig, but got {type(auto_guidance_config)}."
            )
        elif isinstance(next(iter(auto_guidance_config), None), dict):
            auto_guidance_config = [LayerSkipConfig.from_dict(config) for config in auto_guidance_config]

        self.auto_guidance_config = auto_guidance_config
        self._auto_guidance_hook_names = [f"AutoGuidance_{i}" for i in range(len(self.auto_guidance_config))]

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        self._count_prepared += 1
        if self._is_ag_enabled() and self.is_unconditional:
            for name, config in zip(self._auto_guidance_hook_names, self.auto_guidance_config):
                _apply_layer_skip_hook(denoiser, config, name=name)

    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_ag_enabled() and self.is_unconditional:
            for name in self._auto_guidance_hook_names:
                registry = HookRegistry.check_if_exists_or_initialize(denoiser)
                registry.remove_hook(name, recurse=True)

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def prepare_inputs_from_block_state(
        self, data: "BlockState", input_fields: Dict[str, Union[str, Tuple[str, str]]]
    ) -> List["BlockState"]:
        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, self._input_predictions):
            data_batch = self._prepare_batch_from_block_state(input_fields, data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> GuiderOutput:
        pred = None

        if not self._is_ag_enabled():
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_ag_enabled():
            num_conditions += 1
        return num_conditions

    def _is_ag_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self._start * self._num_inference_steps)
            skip_stop_step = int(self._stop * self._num_inference_steps)
            is_within_range = skip_start_step <= self._step < skip_stop_step

        is_close = False
        if self.use_original_formulation:
            is_close = math.isclose(self.guidance_scale, 0.0)
        else:
            is_close = math.isclose(self.guidance_scale, 1.0)

        return is_within_range and not is_close
