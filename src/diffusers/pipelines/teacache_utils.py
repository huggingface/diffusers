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

import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..models import (
    FluxTransformer2DModel,
    HunyuanVideoTransformer3DModel,
    LTXVideoTransformer3DModel,
    LuminaNextDiT2DModel,
    MochiTransformer3DModel,
)
from ..models.hooks import ModelHook, add_hook_to_module
from ..utils import logging
from .pipeline_utils import DiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Source: https://github.com/ali-vilab/TeaCache
# TODO(aryan): Implement methods to calibrate and compute polynomial coefficients on-the-fly, and export to file for re-use.
# fmt: off
_MODEL_TO_POLY_COEFFICIENTS = {
    FluxTransformer2DModel: [4.98651651e02, -2.83781631e02, 5.58554382e01, -3.82021401e00, 2.64230861e-01],
    HunyuanVideoTransformer3DModel: [7.33226126e02, -4.01131952e02, 6.75869174e01, -3.14987800e00, 9.61237896e-02],
    LTXVideoTransformer3DModel: [2.14700694e01, -1.28016453e01, 2.31279151e00, 7.92487521e-01, 9.69274326e-03],
    LuminaNextDiT2DModel: [393.76566581, -603.50993606, 209.10239044, -23.00726601, 0.86377344],
    MochiTransformer3DModel: [-3.51241319e03, 8.11675948e02, -6.09400215e01, 2.42429681e00, 3.05291719e-03],
}
# fmt: on

_MODEL_TO_1_POINT_5X_SPEEDUP_THRESHOLD = {
    FluxTransformer2DModel: 0.25,
    HunyuanVideoTransformer3DModel: 0.1,
    LTXVideoTransformer3DModel: 0.05,
    LuminaNextDiT2DModel: 0.2,
    MochiTransformer3DModel: 0.06,
}

_MODEL_TO_TIMESTEP_MODULATED_LAYER_IDENTIFIER = {
    FluxTransformer2DModel: "transformer_blocks.0.norm1",
}

_MODEL_TO_SKIP_END_LAYER_IDENTIFIER = {
    FluxTransformer2DModel: "norm_out",
}

_DEFAULT_SKIP_LAYER_IDENTIFIERS = [
    "blocks",
    "transformer_blocks",
    "single_transformer_blocks",
    "temporal_transformer_blocks",
]


@dataclass
class TeaCacheConfig:
    l1_threshold: Optional[float] = None

    skip_layer_identifiers: List[str] = _DEFAULT_SKIP_LAYER_IDENTIFIERS

    _polynomial_coefficients: Optional[List[float]] = None


class TeaCacheDenoiserState:
    def __init__(self):
        self.iteration: int = 0
        self.accumulated_l1_difference: float = 0.0
        self.timestep_modulated_cache: torch.Tensor = None
        self.residual_cache: torch.Tensor = None
        self.should_skip_blocks: bool = False

    def reset(self):
        self.iteration = 0
        self.accumulated_l1_difference = 0.0
        self.timestep_modulated_cache = None
        self.residual_cache = None


def apply_teacache(
    pipeline: DiffusionPipeline, config: Optional[TeaCacheConfig] = None, denoiser: Optional[nn.Module] = None
) -> None:
    r"""Applies [TeaCache](https://huggingface.co/papers/2411.19108) to a given pipeline or denoiser module.

    Args:
        TODO
    """

    if config is None:
        logger.warning("No TeaCacheConfig provided. Using default configuration.")
        config = TeaCacheConfig()

    if denoiser is None:
        denoiser = pipeline.transformer if hasattr(pipeline, "transformer") else pipeline.unet

    if isinstance(denoiser, (_MODEL_TO_POLY_COEFFICIENTS.keys())):
        if config.l1_threshold is None:
            logger.info(
                f"No L1 threshold was provided for {type(denoiser)}. Using default threshold as provided in the TeaCache paper for 1.5x speedup. "
                f"For higher speedup, increase the threshold."
            )
            config.l1_threshold = _MODEL_TO_1_POINT_5X_SPEEDUP_THRESHOLD[type(denoiser)]
        if config.timestep_modulated_layer_identifier is None:
            logger.info(
                f"No timestep modulated layer identifier was provided for {type(denoiser)}. Using default identifier as provided in the TeaCache paper."
            )
            config.timestep_modulated_layer_identifier = _MODEL_TO_TIMESTEP_MODULATED_LAYER_IDENTIFIER[type(denoiser)]
        if config._polynomial_coefficients is None:
            logger.info(
                f"No polynomial coefficients were provided for {type(denoiser)}. Using default coefficients as provided in the TeaCache paper."
            )
            config._polynomial_coefficients = _MODEL_TO_POLY_COEFFICIENTS[type(denoiser)]
    else:
        if config.l1_threshold is None:
            raise ValueError(
                f"No L1 threshold was provided for {type(denoiser)}. Using TeaCache with this model is not supported "
                f"in Diffusers. Please provide the L1 threshold in the config by setting the `l1_threshold` attribute."
            )
        if config.timestep_modulated_layer_identifier is None:
            raise ValueError(
                f"No timestep modulated layer identifier was provided for {type(denoiser)}. Using TeaCache with this model is not supported "
                f"in Diffusers. Please provide the layer identifier in the config by setting the `timestep_modulated_layer_identifier` attribute."
            )
        if config._polynomial_coefficients is None:
            raise ValueError(
                f"No polynomial coefficients were provided for {type(denoiser)}. Using TeaCache with this model is not "
                f"supported in Diffusers. Please provide the polynomial coefficients in the config by setting the "
                f"`_polynomial_coefficients` attribute. Automatic calibration will be implemented in the future."
            )

    timestep_modulated_layer_matches = list(
        {
            module
            for name, module in denoiser.named_modules()
            if re.match(config.timestep_modulated_layer_identifier, name)
        }
    )

    if len(timestep_modulated_layer_matches) == 0:
        raise ValueError(
            f"No layer in the denoiser module matched the provided timestep modulated layer identifier: "
            f"{config.timestep_modulated_layer_identifier}. Please provide a valid layer identifier."
        )
    if len(timestep_modulated_layer_matches) > 1:
        logger.warning(
            f"Multiple layers in the denoiser module matched the provided timestep modulated layer identifier: "
            f"{config.timestep_modulated_layer_identifier}. Using the first match."
        )

    denoiser_state = TeaCacheDenoiserState()

    timestep_modulated_layer = timestep_modulated_layer_matches[0]
    hook = TimestepModulatedOutputCacheHook(denoiser_state, config.l1_threshold, config._polynomial_coefficients)
    add_hook_to_module(timestep_modulated_layer, hook, append=True)

    skip_layer_identifiers = config.skip_layer_identifiers
    skip_layer_matches = list(
        {
            module
            for name, module in denoiser.named_modules()
            if any(re.match(identifier, name) for identifier in skip_layer_identifiers)
        }
    )

    for skip_layer in skip_layer_matches:
        hook = DenoiserStateBasedSkipLayerHook(denoiser_state)
        add_hook_to_module(skip_layer, hook, append=True)


class TimestepModulatedOutputCacheHook(ModelHook):
    # The denoiser hook will reset its state, so we don't have to handle it here
    _is_stateful = False

    def __init__(
        self,
        denoiser_state: TeaCacheDenoiserState,
        l1_threshold: float,
        polynomial_coefficients: List[float],
    ) -> None:
        self.denoiser_state = denoiser_state
        self.l1_threshold = l1_threshold
        # TODO(aryan): implement torch equivalent
        self.rescale_fn = np.poly1d(polynomial_coefficients)

    def post_forward(self, module, output):
        if isinstance(output, tuple):
            # This assumes that the first element of the output tuple is the timestep modulated noise output.
            # For Diffusers models, this is true. For models outside diffusers, users will have to ensure
            # that the first element of the output tuple is the timestep modulated noise output (seems to be
            # the case for most research model implementations).
            timestep_modulated_noise = output[0]
        elif torch.is_tensor(output):
            timestep_modulated_noise = output
        else:
            raise ValueError(
                f"Expected output to be a tensor or a tuple with first element as timestep modulated noise. "
                f"Got {type(output)} instead. Please ensure that the denoiser module returns the timestep "
                f"modulated noise output as the first element."
            )

        if self.denoiser_state.timestep_modulated_cache is not None:
            l1_diff = (timestep_modulated_noise - self.denoiser_state.timestep_modulated_cache).abs().mean()
            normalized_l1_diff = l1_diff / self.denoiser_state.timestep_modulated_cache.abs().mean()
            rescaled_l1_diff = self.rescale_fn(normalized_l1_diff)
            self.denoiser_state.accumulated_l1_difference += rescaled_l1_diff

            if self.denoiser_state.accumulated_l1_difference >= self.l1_threshold:
                self.denoiser_state.should_skip_blocks = True
                self.denoiser_state.accumulated_l1_difference = 0.0
            else:
                self.denoiser_state.should_skip_blocks = False

        self.denoiser_state.timestep_modulated_cache = timestep_modulated_noise
        return output


class DenoiserStateBasedSkipLayerHook(ModelHook):
    _is_stateful = False

    def __init__(self, denoiser_state: TeaCacheDenoiserState) -> None:
        self.denoiser_state = denoiser_state

    def new_forward(self, module, *args, **kwargs):
        args, kwargs = module._diffusers_hook.pre_forward(module, *args, **kwargs)

        if not self.denoiser_state.should_skip_blocks:
            output = module._old_forward(*args, **kwargs)
        else:
            # Diffusers models either expect one output (hidden_states) or a tuple of two outputs (hidden_states, encoder_hidden_states).
            # Returning a tuple of None values handles both cases. It is okay to do because we are not going to be using these
            # anywhere if self.denoiser_state.should_skip_blocks is True.
            output = (None, None)

        return module._diffusers_hook.post_forward(module, output)
