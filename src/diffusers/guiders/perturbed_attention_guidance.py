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

from .guider_utils import GuidanceMixin, _replace_attention_processors, rescale_noise_cfg


class PerturbedAttentionGuidance(GuidanceMixin):
    """
    Perturbed Attention Guidance (PAB): https://huggingface.co/papers/2403.17377

    Args:
        pag_applied_layers (`str` or `List[str]`):
            The name of the attention layers where Perturbed Attention Guidance is applied. This can be a single layer
            name or a list of layer names. The names should either be FQNs (fully qualified names) to each attention
            layer or a regex pattern that matches the FQNs of the attention layers. For example, if you want to apply
            PAG to transformer blocks 10 and 20, you can set this to `["transformer_blocks.10",
            "transformer_blocks.20"]`, or `"transformer_blocks.(10|20)"`.
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        pag_scale (`float`, defaults to `3.0`):
            The scale parameter for perturbed attention guidance.
        guidance_rescale (`float`, defaults to `0.0`):
            The rescale factor applied to the noise predictions. This is used to improve image quality and fix
            overexposure. Based on Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
            Flawed](https://huggingface.co/papers/2305.08891).
        use_original_formulation (`bool`, defaults to `False`):
            Whether to use the original formulation of classifier-free guidance as proposed in the paper. By default,
            we use the diffusers-native implementation that has been in the codebase for a long time. See
            [~guiders.classifier_free_guidance.ClassifierFreeGuidance] for more details.
    """

    _input_predictions = ["pred_cond", "pred_uncond", "pred_perturbed"]

    def __init__(
        self,
        pag_applied_layers: Union[str, List[str]],
        guidance_scale: float = 7.5,
        pag_scale: float = 3.0,
        skip_context_attention: bool = False,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
    ):
        super().__init__()

        self.pag_applied_layers = pag_applied_layers
        self.guidance_scale = guidance_scale
        self.pag_scale = pag_scale
        self.skip_context_attention = skip_context_attention
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        self._is_pag_batch = False
        self._original_processors = None
        self._denoiser = None

    def prepare_models(self, denoiser: torch.nn.Module):
        self._denoiser = denoiser

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

    def prepare_outputs(self, pred: torch.Tensor) -> None:
        self._num_outputs_prepared += 1
        if self._num_outputs_prepared > self.num_conditions:
            raise ValueError(f"Expected {self.num_conditions} outputs, but prepare_outputs called more times.")
        key = self._input_predictions[self._num_outputs_prepared - 1]
        if not self._is_cfg_enabled() and self._is_pag_enabled():
            # If we're predicting pred_cond and pred_perturbed only, we need to set the key to pred_perturbed
            # to avoid writing into pred_uncond which is not used
            if self._num_outputs_prepared == 2:
                key = "pred_perturbed"
        self._preds[key] = pred

        # Prepare denoiser for perturbed attention prediction if needed
        if not self._is_pag_enabled():
            return
        should_register_pag = (self._is_cfg_enabled() and self._num_outputs_prepared == 2) or (
            not self._is_cfg_enabled() and self._num_outputs_prepared == 1
        )
        if should_register_pag:
            self._is_pag_batch = True
            self._original_processors = _replace_attention_processors(
                self._denoiser,
                self.pag_applied_layers,
                skip_context_attention=self.skip_context_attention,
                metadata_name="perturbed_attention_guidance_processor_cls",
            )
        elif self._is_pag_batch:
            # Restore the original attention processors
            _replace_attention_processors(self._denoiser, processors=self._original_processors)
            self._is_pag_batch = False
            self._original_processors = None

    def cleanup_models(self, denoiser: torch.nn.Module):
        self._denoiser = None

    def forward(
        self,
        pred_cond: torch.Tensor,
        pred_uncond: Optional[torch.Tensor] = None,
        pred_perturbed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred = None

        if not self._is_cfg_enabled() and not self._is_pag_enabled():
            pred = pred_cond
        elif not self._is_cfg_enabled():
            shift = pred_cond - pred_perturbed
            pred = pred_cond + self.pag_scale * shift
        elif not self._is_pag_enabled():
            shift = pred_cond - pred_uncond
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift
        else:
            shift = pred_cond - pred_uncond
            shift_perturbed = pred_cond - pred_perturbed
            pred = pred_cond if self.use_original_formulation else pred_uncond
            pred = pred + self.guidance_scale * shift + self.pag_scale * shift_perturbed

        if self.guidance_rescale > 0.0:
            pred = rescale_noise_cfg(pred, pred_cond, self.guidance_rescale)

        return pred

    @property
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_pag_enabled():
            num_conditions += 1
        return num_conditions

    def _is_cfg_enabled(self) -> bool:
        if self.use_original_formulation:
            return not math.isclose(self.guidance_scale, 0.0)
        else:
            return not math.isclose(self.guidance_scale, 1.0)

    def _is_pag_enabled(self) -> bool:
        is_zero = math.isclose(self.pag_scale, 0.0)
        return not is_zero
