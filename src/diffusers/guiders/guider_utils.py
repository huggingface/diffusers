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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch

from ..utils import deprecate, get_logger


if TYPE_CHECKING:
    from ..models.attention_processor import AttentionProcessor


logger = get_logger(__name__)  # pylint: disable=invalid-name


class GuidanceMixin:
    r"""Base mixin class providing the skeleton for implementing guidance techniques."""

    _input_predictions = None

    def __init__(self):
        self._step: int = None
        self._num_inference_steps: int = None
        self._timestep: torch.LongTensor = None
        self._preds: Dict[str, torch.Tensor] = {}
        self._num_outputs_prepared: int = 0

        if self._input_predictions is None or not isinstance(self._input_predictions, list):
            raise ValueError(
                "`_input_predictions` must be a list of required prediction names for the guidance technique."
            )

    def set_state(self, step: int, num_inference_steps: int, timestep: torch.LongTensor) -> None:
        self._step = step
        self._num_inference_steps = num_inference_steps
        self._timestep = timestep
        self._preds = {}
        self._num_outputs_prepared = 0

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        pass

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
                    # Alternating conditional and unconditional inputs as batches
                    inputs = [arg[i % 2] for i in range(num_conditions)]
                    list_of_inputs.append(inputs)
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
        self._preds[key] = pred

    def cleanup_models(self, denoiser: torch.nn.Module) -> None:
        pass

    def __call__(self, **kwargs) -> Any:
        if len(kwargs) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} arguments, but got {len(kwargs)}. Please provide the correct number of arguments."
            )
        return self.forward(**kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("GuidanceMixin::forward must be implemented in subclasses.")

    @property
    def num_conditions(self) -> int:
        raise NotImplementedError("GuidanceMixin::num_conditions must be implemented in subclasses.")

    @property
    def outputs(self) -> Dict[str, torch.Tensor]:
        return self._preds


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


def _replace_attention_processors(
    module: torch.nn.Module,
    pag_applied_layers: Optional[Union[str, List[str]]] = None,
    skip_context_attention: bool = False,
    processors: Optional[List[Tuple[torch.nn.Module, "AttentionProcessor"]]] = None,
    metadata_name: Optional[str] = None,
) -> Optional[List[Tuple[torch.nn.Module, "AttentionProcessor"]]]:
    if processors is not None and metadata_name is not None:
        raise ValueError("Cannot pass both `processors` and `metadata_name` at the same time.")
    if metadata_name is not None:
        if isinstance(pag_applied_layers, str):
            pag_applied_layers = [pag_applied_layers]
        return _replace_layers_with_guidance_processors(
            module, pag_applied_layers, skip_context_attention, metadata_name
        )
    if processors is not None:
        _replace_layers_with_existing_processors(processors)


def _replace_layers_with_guidance_processors(
    module: torch.nn.Module,
    pag_applied_layers: List[str],
    skip_context_attention: bool,
    metadata_name: str,
) -> List[Tuple[torch.nn.Module, "AttentionProcessor"]]:
    from ..hooks._common import _ATTENTION_CLASSES
    from ..hooks._helpers import GuidanceMetadataRegistry

    processors = []
    for name, submodule in module.named_modules():
        if (
            (not isinstance(submodule, _ATTENTION_CLASSES))
            or (getattr(submodule, "processor", None) is None)
            or not (
                any(
                    re.search(pag_layer, name) is not None and not _is_fake_integral_match(pag_layer, name)
                    for pag_layer in pag_applied_layers
                )
            )
        ):
            continue
        old_attention_processor = submodule.processor
        metadata = GuidanceMetadataRegistry.get(old_attention_processor.__class__)
        new_attention_processor_cls = getattr(metadata, metadata_name)
        new_attention_processor = new_attention_processor_cls()
        # !!! dunder methods cannot be replaced on instances !!!
        # if "skip_context_attention" in inspect.signature(new_attention_processor.__call__).parameters:
        #     new_attention_processor.__call__ = partial(
        #         new_attention_processor.__call__, skip_context_attention=skip_context_attention
        #     )
        submodule.processor = new_attention_processor
        processors.append((submodule, old_attention_processor))
    return processors


def _replace_layers_with_existing_processors(processors: List[Tuple[torch.nn.Module, "AttentionProcessor"]]) -> None:
    for module, proc in processors:
        module.processor = proc


def _is_fake_integral_match(layer_id, name):
    layer_id = layer_id.split(".")[-1]
    name = name.split(".")[-1]
    return layer_id.isnumeric() and name.isnumeric() and layer_id == name


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
