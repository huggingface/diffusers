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
from .guider_utils import GuidanceMixin, rescale_noise_cfg


class SkipLayerGuidance(GuidanceMixin):
    """
    Skip Layer Guidance (SLG): https://github.com/Stability-AI/sd3.5

    CFG is a technique used to improve generation quality and condition-following in diffusion models. It works by
    jointly training a model on both conditional and unconditional data, and using a weighted sum of the two during
    inference. This allows the model to tradeoff between generation quality and sample diversity.

    The original paper proposes scaling and shifting the conditional distribution based on the difference between
    conditional and unconditional predictions. [x_pred = x_cond + scale * (x_cond - x_uncond)]

    Diffusers implemented the scaling and shifting on the unconditional prediction instead, which is equivalent to what
    the original paper proposed in theory. [x_pred = x_uncond + scale * (x_cond - x_uncond)]

    The intution behind the original formulation can be thought of as moving the conditional distribution estimates
    further away from the unconditional distribution estimates, while the diffusers-native implementation can be
    thought of as moving the unconditional distribution towards the conditional distribution estimates to get rid of
    the unconditional predictions (usually negative features like "bad quality, bad anotomy, watermarks", etc.)

    The `use_original_formulation` argument can be set to `True` to use the original CFG formulation mentioned in the
    paper. By default, we use the diffusers-native implementation that has been in the codebase for a long time.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time.
    """

    def __init__(
        self,
        guidance_scale: float = 7.5,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_start: float = 0.01,
        skip_layer_guidance_stop: float = 0.2,
        skip_guidance_layers: Optional[Union[int, List[int]]] = None,
        skip_layer_config: Union[LayerSkipConfig, List[LayerSkipConfig]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
    ):
        self.guidance_scale = guidance_scale
        self.skip_layer_guidance_scale = skip_layer_guidance_scale
        self.skip_layer_guidance_start = skip_layer_guidance_start
        self.skip_layer_guidance_stop = skip_layer_guidance_stop
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if skip_guidance_layers is None and skip_layer_config is None:
            raise ValueError(
                "Either `skip_guidance_layers` or `skip_layer_config` must be provided to enable Skip Layer Guidance."
            )
        if skip_guidance_layers is not None and skip_layer_config is not None:
            raise ValueError("Only one of `skip_guidance_layers` or `skip_layer_config` can be provided.")

        if skip_guidance_layers is not None:
            if isinstance(skip_guidance_layers, int):
                skip_guidance_layers = [skip_guidance_layers]
            if not isinstance(skip_guidance_layers, list):
                raise ValueError(
                    f"Expected `skip_guidance_layers` to be an int or a list of ints, but got {type(skip_guidance_layers)}."
                )
            skip_layer_config = [LayerSkipConfig(layer, fqn="auto") for layer in skip_guidance_layers]

        if isinstance(skip_layer_config, LayerSkipConfig):
            skip_layer_config = [skip_layer_config]

        if not isinstance(skip_layer_config, list):
            raise ValueError(
                f"Expected `skip_layer_config` to be a LayerSkipConfig or a list of LayerSkipConfig, but got {type(skip_layer_config)}."
            )

        self.skip_layer_config = skip_layer_config
        self._skip_layer_hook_names = [f"SkipLayerGuidance_{i}" for i in range(len(self.skip_layer_config))]

    def prepare_models(self, denoiser: torch.nn.Module):
        skip_start_step = int(self.skip_layer_guidance_start * self._num_inference_steps)
        skip_stop_step = int(self.skip_layer_guidance_stop * self._num_inference_steps)

        # Register the hooks for layer skipping if the step is within the specified range
        if skip_start_step < self._step < skip_stop_step:
            for name, config in zip(self._skip_layer_hook_names, self.skip_layer_config):
                _apply_layer_skip_hook(denoiser, config, name=name)

    def prepare_inputs(self, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        num_conditions = self.num_conditions
        list_of_inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
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

    def cleanup_models(self, denoiser: torch.nn.Module):
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
        skip_start_step = int(self.skip_layer_guidance_start * self._num_inference_steps)
        skip_stop_step = int(self.skip_layer_guidance_stop * self._num_inference_steps)

        if math.isclose(self.guidance_scale, 1.0) and math.isclose(self.skip_layer_guidance_scale, 1.0):
            pred = pred_cond

        elif math.isclose(self.guidance_scale, 1.0):
            if skip_start_step < self._step < skip_stop_step:
                shift = pred_cond - pred_cond_skip
                pred = pred_cond if self.use_original_formulation else pred_cond_skip
                pred = pred + self.skip_layer_guidance_scale * shift
            else:
                pred = pred_cond

        elif math.isclose(self.skip_layer_guidance_scale, 1.0):
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

            if skip_start_step < self._step < skip_stop_step:
                shift_skip = pred_cond - pred_cond_skip
                pred = pred + self.skip_layer_guidance_scale * shift_skip

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        skip_start_step = int(self.skip_layer_guidance_start * self._num_inference_steps)
        skip_stop_step = int(self.skip_layer_guidance_stop * self._num_inference_steps)

        if not math.isclose(self.guidance_scale, 1.0):
            num_conditions += 1
        if not math.isclose(self.skip_layer_guidance_scale, 1.0) and skip_start_step < self._step < skip_stop_step:
            num_conditions += 1

        return num_conditions
