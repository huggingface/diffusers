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
from ..utils import get_logger
from .guider_utils import BaseGuidance, GuiderOutput, rescale_noise_cfg


if TYPE_CHECKING:
    from ..modular_pipelines.modular_pipeline import BlockState


logger = get_logger(__name__)  # pylint: disable=invalid-name


class PerturbedAttentionGuidance(BaseGuidance):
    """
    Perturbed Attention Guidance (PAG): https://huggingface.co/papers/2403.17377

    The intution behind PAG can be thought of as moving the CFG predicted distribution estimates further away from
    worse versions of the conditional distribution estimates. PAG was one of the first techniques to introduce the idea
    of using a worse version of the trained model for better guiding itself in the denoising process. It perturbs the
    attention scores of the latent stream by replacing the score matrix with an identity matrix for selectively chosen
    layers.

    Additional reading:
    - [Guiding a Diffusion Model with a Bad Version of Itself](https://huggingface.co/papers/2406.02507)

    PAG is implemented with similar implementation to SkipLayerGuidance due to overlap in the configuration parameters
    and implementation details.

    Args:
        guidance_scale (`float`, defaults to `7.5`):
            The scale parameter for classifier-free guidance. Higher values result in stronger conditioning on the text
            prompt, while lower values allow for more freedom in generation. Higher values may lead to saturation and
            deterioration of image quality.
        perturbed_guidance_scale (`float`, defaults to `2.8`):
            The scale parameter for perturbed attention guidance.
        perturbed_guidance_start (`float`, defaults to `0.01`):
            The fraction of the total number of denoising steps after which perturbed attention guidance starts.
        perturbed_guidance_stop (`float`, defaults to `0.2`):
            The fraction of the total number of denoising steps after which perturbed attention guidance stops.
        perturbed_guidance_layers (`int` or `List[int]`, *optional*):
            The layer indices to apply perturbed attention guidance to. Can be a single integer or a list of integers.
            If not provided, `perturbed_guidance_config` must be provided.
        perturbed_guidance_config (`LayerSkipConfig` or `List[LayerSkipConfig]`, *optional*):
            The configuration for the perturbed attention guidance. Can be a single `LayerSkipConfig` or a list of
            `LayerSkipConfig`. If not provided, `perturbed_guidance_layers` must be provided.
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

    # NOTE: The current implementation does not account for joint latent conditioning (text + image/video tokens in
    # the same latent stream). It assumes the entire latent is a single stream of visual tokens. It would be very
    # complex to support joint latent conditioning in a model-agnostic manner without specializing the implementation
    # for each model architecture.

    _input_predictions = ["pred_cond", "pred_uncond", "pred_cond_skip"]

    @register_to_config
    def __init__(
        self,
        guidance_scale: float = 7.5,
        perturbed_guidance_scale: float = 2.8,
        perturbed_guidance_start: float = 0.01,
        perturbed_guidance_stop: float = 0.2,
        perturbed_guidance_layers: Optional[Union[int, List[int]]] = None,
        perturbed_guidance_config: Union[LayerSkipConfig, List[LayerSkipConfig], Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__(start, stop, enabled)

        self.guidance_scale = guidance_scale
        self.skip_layer_guidance_scale = perturbed_guidance_scale
        self.skip_layer_guidance_start = perturbed_guidance_start
        self.skip_layer_guidance_stop = perturbed_guidance_stop
        self.guidance_rescale = guidance_rescale
        self.use_original_formulation = use_original_formulation

        if perturbed_guidance_config is None:
            if perturbed_guidance_layers is None:
                raise ValueError(
                    "`perturbed_guidance_layers` must be provided if `perturbed_guidance_config` is not specified."
                )
            perturbed_guidance_config = LayerSkipConfig(
                indices=perturbed_guidance_layers,
                fqn="auto",
                skip_attention=False,
                skip_attention_scores=True,
                skip_ff=False,
            )
        else:
            if perturbed_guidance_layers is not None:
                raise ValueError(
                    "`perturbed_guidance_layers` should not be provided if `perturbed_guidance_config` is specified."
                )

        if isinstance(perturbed_guidance_config, dict):
            perturbed_guidance_config = LayerSkipConfig.from_dict(perturbed_guidance_config)

        if isinstance(perturbed_guidance_config, LayerSkipConfig):
            perturbed_guidance_config = [perturbed_guidance_config]

        if not isinstance(perturbed_guidance_config, list):
            raise ValueError(
                "`perturbed_guidance_config` must be a `LayerSkipConfig`, a list of `LayerSkipConfig`, or a dict that can be converted to a `LayerSkipConfig`."
            )
        elif isinstance(next(iter(perturbed_guidance_config), None), dict):
            perturbed_guidance_config = [LayerSkipConfig.from_dict(config) for config in perturbed_guidance_config]

        for config in perturbed_guidance_config:
            if config.skip_attention or not config.skip_attention_scores or config.skip_ff:
                logger.warning(
                    "Perturbed Attention Guidance is designed to perturb attention scores, so `skip_attention` should be False, `skip_attention_scores` should be True, and `skip_ff` should be False. "
                    "Please check your configuration. Modifying the config to match the expected values."
                )
            config.skip_attention = False
            config.skip_attention_scores = True
            config.skip_ff = False

        self.skip_layer_config = perturbed_guidance_config
        self._skip_layer_hook_names = [f"SkipLayerGuidance_{i}" for i in range(len(self.skip_layer_config))]

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.prepare_models
    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        self._count_prepared += 1
        if self._is_slg_enabled() and self.is_conditional and self._count_prepared > 1:
            for name, config in zip(self._skip_layer_hook_names, self.skip_layer_config):
                _apply_layer_skip_hook(denoiser, config, name=name)

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.cleanup_models
    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        if self._is_slg_enabled() and self.is_conditional and self._count_prepared > 1:
            registry = HookRegistry.check_if_exists_or_initialize(denoiser)
            # Remove the hooks after inference
            for hook_name in self._skip_layer_hook_names:
                registry.remove_hook(hook_name, recurse=True)

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.prepare_inputs
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

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.forward
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
    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.is_conditional
    def is_conditional(self) -> bool:
        return self._count_prepared == 1 or self._count_prepared == 3

    @property
    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance.num_conditions
    def num_conditions(self) -> int:
        num_conditions = 1
        if self._is_cfg_enabled():
            num_conditions += 1
        if self._is_slg_enabled():
            num_conditions += 1
        return num_conditions

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance._is_cfg_enabled
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

    # Copied from diffusers.guiders.skip_layer_guidance.SkipLayerGuidance._is_slg_enabled
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
