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


class SkipLayerGuidance(BaseGuidance):
    """
    Skip Layer Guidance (SLG): https://github.com/Stability-AI/sd3.5

    Spatio-Temporal Guidance (STG): https://huggingface.co/papers/2411.18664

    SLG was introduced by StabilityAI for improving structure and anotomy coherence in generated images. It works by
    skipping the forward pass of specified transformer blocks during the denoising process on an additional conditional
    batch of data, apart from the conditional and unconditional batches already used in CFG
    ([~guiders.classifier_free_guidance.ClassifierFreeGuidance]), and then scaling and shifting the CFG predictions
    based on the difference between conditional without skipping and conditional with skipping predictions.

    The intution behind SLG can be thought of as moving the CFG predicted distribution estimates further away from
    worse versions of the conditional distribution estimates (because skipping layers is equivalent to using a worse
    version of the model for the conditional prediction).

    STG is an improvement and follow-up work combining ideas from SLG, PAG and similar techniques for improving
    generation quality in video diffusion models.

    Additional reading:
    - [Guiding a Diffusion Model with a Bad Version of Itself](https://huggingface.co/papers/2406.02507)

    The values for `skip_layer_guidance_scale`, `skip_layer_guidance_start`, and `skip_layer_guidance_stop` are
    defaulted to the recommendations by StabilityAI for Stable Diffusion 3.5 Medium.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        skip_layer_guidance_scale (`float`, defaults to `2.8`):
            The scale parameter for skip layer guidance. Anatomy and structure coherence may improve with higher
            values, but it may also lead to overexposure and saturation.
        skip_layer_guidance_start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which skip layer guidance starts.
        skip_layer_guidance_stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which skip layer guidance stops.
        skip_layer_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply skip layer guidance to. Can be a single integer or a list of integers. If not
            provided, `skip_layer_config` must be provided. The recommended values are `[7, 8, 9]` for Stable Diffusion
            3.5 Medium.
        skip_layer_config (`LayerSkipConfig` or `List[LayerSkipConfig]`, *optional*):
            The configuration for the skip layer guidance. Can be a single `LayerSkipConfig` or a list of
            `LayerSkipConfig`. If not provided, `skip_layer_guidance_layers` must be provided.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
        start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which guidance starts.
        stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which guidance stops.
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_skip"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_start: float = 0.01,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_layers: Optional[Union[int, List[int]]] = None,
        skip_layer_config: Union[LayerSkipConfig, List[LayerSkipConfig], Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.skip_layer_guidance_scale = skip_layer_guidance_scale
        self.skip_layer_guidance_start = skip_layer_guidance_start
        self.skip_layer_guidance_stop = skip_layer_guidance_stop
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if not (0.0 <= skip_layer_guidance_start < 1.0):
            raise ValueError(
                f"Expected `skip_layer_guidance_start` to be between 0.0 and 1.0, but got {skip_layer_guidance_start}."
            )
        if not (skip_layer_guidance_start <= skip_layer_guidance_stop <= 1.0):
            raise ValueError(
                f"Expected `skip_layer_guidance_stop` to be between 0.0 and 1.0, but got {skip_layer_guidance_stop}."
            )

        if skip_layer_guidance_layers is None and skip_layer_config is None:
            raise ValueError(
                "Either `skip_layer_guidance_layers` or `skip_layer_config` must be provided to enable Skip Layer Guidance."
            )
        if skip_layer_guidance_layers is not None and skip_layer_config is not None:
            raise ValueError("Only one of `skip_layer_guidance_layers` or `skip_layer_config` can be provided.")

        if skip_layer_guidance_layers is not None:
            if isinstance(skip_layer_guidance_layers, int):
                skip_layer_guidance_layers = [skip_layer_guidance_layers]
            if not isinstance(skip_layer_guidance_layers, list):
                raise ValueError(
                    f"Expected `skip_layer_guidance_layers` to be an int or a list of ints, but got {type(skip_layer_guidance_layers)}."
                )
            skip_layer_config = [LayerSkipConfig(layer, fqn="auto") for layer in skip_layer_guidance_layers]

        if isinstance(skip_layer_config, dict):
            skip_layer_config = LayerSkipConfig.from_dict(skip_layer_config)

        if isinstance(skip_layer_config, LayerSkipConfig):
            skip_layer_config = [skip_layer_config]

        if not isinstance(skip_layer_config, list):
            raise ValueError(
                f"Expected `skip_layer_config` to be a LayerSkipConfig or a list of LayerSkipConfig, but got {type(skip_layer_config)}."
            )
        elif isinstance(next(iter(skip_layer_config), None), dict):
            skip_layer_config = [LayerSkipConfig.from_dict(config) for config in skip_layer_config]

        self.skip_layer_config = skip_layer_config
        self._skip_layer_hook_names = [f"SkipLayerGuidance_{i}" for i in range(len(self.skip_layer_config))]

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        self._count_prepared += 1
        if self._is_slg_enabled() and self.is_conditional and self._count_prepared > 1:
            for name, config in zip(self._skip_layer_hook_names, self.skip_layer_config):
                _apply_layer_skip_hook(denoiser, config, name=name)

    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_slg_enabled() and self.is_conditional and self._count_prepared > 1:
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._skip_layer_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    def prepare_inputs(self, data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> List["BlockState"]:
        if self.num_conditions == 1:
            tuple_indices = [0]
            input_predictions = ["pred_cond"]
        elif self.num_conditions == 2:
            tuple_indices = [0, 1]
            input_predictions = (
                ["pred_cond", "pred_uncond"] if self._is_cfg_enabled() else ["pred_cond", "pred_cond_skip"]
            )
        else:
            tuple_indices = [0, 1, 0]
            input_predictions = ["pred_cond", "pred_uncond", "pred_cond_skip"]
        data_batches = []
        for tuple_idx, input_prediction in zip(tuple_indices, input_predictions):
            data_batch = self._prepare_batch(data, tuple_idx, input_prediction)
            data_batches.append(data_batch)
        return data_batches

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        pred_cond_skip: Optional[torch.Tensor] = None,
    ) -> GuiderOutput:
        pred = None

        if not self._is_cfg_enabled() and not self._is_slg_enabled():
            pred = pred_cond
        elif not self._is_cfg_enabled():
            shift = pred_cond - pred_cond_skip
            pred = pred_cond if self.use_original_formulation else pred_cond_skip
            pred = pred + self.skip_layer_guidance_scale * shift
        elif not self._is_slg_enabled():
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift
        else:
            shift = pred_cond - pred_uncond
            shift_skip = pred_cond - pred_cond_skip
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift + self.skip_layer_guidance_scale * shift_skip

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return GuiderOutput(pred=pred, pred_cond=pred_cond, pred_uncond=pred_uncond)

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1 or self._count_prepared == 3

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_slg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
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

    def _is_slg_enabled(self) -> bool:
        if not self._enabled:
            return False

        is_within_range = True
        if self._num_inference_steps is not None:
            skip_start_step = int(self.skip_layer_guidance_start * self._num_inference_steps)
            skip_stop_step = int(self.skip_layer_guidance_stop * self._num_inference_steps)
            is_within_range = skip_start_step < self._step < skip_stop_step

        is_zero = math.isclose(self.skip_layer_guidance_scale, 0.0)

        return is_within_range and not is_zero
