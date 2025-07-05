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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from ..configuration_utils import register_to_config
from .guider_utils import BaseGuidance, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


class ClassifierFreeGuidance(BaseGuidance):
    """
    Classifier-free guidance (CFG): https://huggingface.co/papers/2207.12598

    CFG is a technique used to improve generation quality and condition-following in diffusion models. It works by
    jointly training a model on both conditional and unconditional data, and using a weighted sum of the two during
    inference. This allows the model to tradeoff between generation quality and sample diversity. The original paper
    proposes scaling and shifting the conditional distribution based on the difference between conditional and
    unconditional predictions. [x_pred = x_cond + scale * (x_cond - x_uncond)]

    Diffusers implemented the scaling and shifting on the unconditional prediction instead based on the [Imagen
    paper](https://huggingface.co/papers/2205.11487), which is equivalent to what the original paper proposed in
    theory. [x_pred = x_uncond + scale * (x_cond - x_uncond)]

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
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__(start, stop)

        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

    def prepare_inputs(
        self, data: "BlockState", input_fields: Optional[Dict[str, Union[str, Tuple[str, str]]]] = None
    ) -> List["BlockState"]:
        if input_fields is None:
            input_fields = self._input_fields

        tuple_indices = [0] if self.num_conditions == 1 else [0, 1]
        data_batches = []
        for i in range(self.num_conditions):
            data_batch = self._prepare_batch(input_fields, data, tuple_indices[i], self._input_predictions[i])
            data_batches.append(data_batch)
        return data_batches

    def forward(self, pred_cond: torch.Tensor, pred_uncond: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = None

        if not self._is_cfg_enabled():
            pred = pred_cond
        else:
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred, {}

    @property
    def is_conditional(self) -> bool:
        return self._count_prepared == 1

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
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
