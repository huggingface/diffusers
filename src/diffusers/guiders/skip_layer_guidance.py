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

import math
from typing import List, Optional, Tuple, Union

import torch

from ..hooks import HookRegistry, LayerSkipConfig
from ..hooks.layer_skip import _apply_layer_skip_hook
from .guider_utils import BaseGuidance, rescale_noise_cfg


class SkipLayerGuidance(BaseGuidance):
    """
    Skip Layer Guidance (SLG): https://github.com/Stability-AI/sd3.5 Spatio-Temporal Guidance (STG):
    https://huggingface.co/papers/2411.18664
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

    def __init__(
        self,
        guidance_scale: float = 7.5,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_layers: Optional[Union[int, List[int]]] = None,
        skip_layer_config: Union[LayerSkipConfig, List[LayerSkipConfig]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.01,
        stop: float = 0.2,
    ):
        super().__init__(start, stop)

        self.guidance_scale = guidance_scale
        self.skip_layer_guidance_scale = skip_layer_guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

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

        if isinstance(skip_layer_config, LayerSkipConfig):
            skip_layer_config = [skip_layer_config]

        if not isinstance(skip_layer_config, list):
            raise ValueError(
                f"Expected `skip_layer_config` to be a LayerSkipConfig or a list of LayerSkipConfig, but got {type(skip_layer_config)}."
            )

        self.skip_layer_config = skip_layer_config
        self._skip_layer_hook_names = [f"SkipLayerGuidance_{i}" for i in range(len(self.skip_layer_config))]

    def prepare_inputs(self, denoiser: torch.nn.Module, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        if self._num_outputs_prepared == 0 and self._is_slg_enabled():
            for name, config in zip(self._skip_layer_hook_names, self.skip_layer_config):
                _apply_layer_skip_hook(denoiser, config, name=name)
        
        num_conditions = self.num_conditions
        list_of_inputs = []
        for arg in args:
            if arg is None or isinstance(arg, torch.Tensor):
                list_of_inputs.append([arg] * num_conditions)
            elif isinstance(arg, (tuple, list)):
                if len(arg) != 2:
                    raise ValueError(
                        f"Expected a tuple or list of length 2, but got {len(arg)} for argument {arg}. Please provide a tuple/list of length 2 "
                        f"with the first element being the conditional input and the second element being the unconditional input or None."
                    )
                if arg[1] is None:
                    # Only conditioning inputs for all batches
                    list_of_inputs.append([arg[0]] * num_conditions)
                else:
                    list_of_inputs.append([arg[0], arg[1], arg[0]])
            else:
                raise ValueError(
                    f"Expected a tensor, tuple, or list, but got {type(arg)} for argument {arg}. Please provide a tensor, tuple, or list."
                )
        return tuple(list_of_inputs)

    def prepare_outputs(self, denoiser: torch.nn.Module, pred: torch.Tensor) -> None:
        self._num_outputs_prepared += 1
        if self._num_outputs_prepared > self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} outputs, but prepare_outputs called more times.")
        key = self._input_predictions[self._num_outputs_prepared - 1]
        if not self._is_cfg_enabled() and self._is_slg_enabled():
            # If we're predicting pred_cond and pred_cond_skip only, we need to set the key to pred_cond_skip
            # to avoid writing into pred_uncond which is not used
            if self._num_outputs_prepared == 2:
                key = "pred_cond_skip"
        self._preds[key] = pred

        if self._num_outputs_prepared == self.num_conditions:
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._skip_layer_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        pred_cond_skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_slg_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
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
        skip_start_step = int(self._start * self._num_inference_steps)
        skip_stop_step = int(self._stop * self._num_inference_steps)
        is_within_range = skip_start_step < self._step < skip_stop_step
        is_zero = math.isclose(self.skip_layer_guidance_scale, 0.0)
        return is_within_range and not is_zero
