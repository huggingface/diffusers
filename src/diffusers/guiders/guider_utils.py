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

from typing import Any, List, Optional, Tuple, Union

import torch

from ..utils import deprecate, get_logger


logger = get_logger(__name__)  # pylint: disable=invalid-name


class GuidanceMixin:
    r"""Base mixin class providing the skeleton for implementing guidance techniques."""

    def prepare_inputs(self, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        num_conditions = self.num_conditions
        list_of_inputs = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                list_of_inputs.append([arg] * num_conditions)
            elif isinstance(arg, (tuple, list)):
                inputs = [x for x in arg if x is not None]
                if len(inputs) < num_conditions:
                    raise ValueError(f"Required at least {num_conditions} inputs, but got {len(inputs)}.")
                list_of_inputs.append(inputs[:num_conditions])
            else:
                raise ValueError(
                    f"Expected a tensor, tuple, or list, but got {type(arg)} for argument {arg}. Please provide a tensor, tuple, or list."
                )
        return tuple(list_of_inputs)

    def __call__(self, *args) -> Any:
        if len(args) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} arguments, but got {len(args)}. Please provide the correct number of arguments."
            )
        return self.forward(*args)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("GuidanceMixin::forward must be implemented in subclasses.")

    @property
    def num_conditions(self) -> int:
        raise NotImplementedError("GuidanceMixin::num_conditions must be implemented in subclasses.")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.

    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def _raise_guidance_deprecation_warning(
    *,
    guidance_scale: Optional[float] = None,
    guidance_rescale: Optional[float] = None,
) -> None:
    if guidance_scale is not None:
        msg = "The `guidance_scale` argument is deprecated and will be removed in version 1.0.0. Please pass a `GuidanceMixin` object for the `guidance` argument instead."
        deprecate("guidance_scale", "1.0.0", msg, standard_warn=False)
    if guidance_rescale is not None:
        msg = "The `guidance_rescale` argument is deprecated and will be removed in version 1.0.0. Please pass a `GuidanceMixin` object for the `guidance` argument instead."
        deprecate("guidance_rescale", "1.0.0", msg, standard_warn=False)
