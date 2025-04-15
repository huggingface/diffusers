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

from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import torch

from ..utils import get_logger


if TYPE_CHECKING:
    from ..models.attention_processor import AttentionProcessor


logger = get_logger(__name__)  # pylint: disable=invalid-name


class BaseGuidance:
    r"""Base class providing the skeleton for implementing guidance techniques."""

    _input_predictions = None

    def __init__(self, start: float = 0.0, stop: float = 1.0):
        self._start = start
        self._stop = stop
        self._step: int = None
        self._num_inference_steps: int = None
        self._timestep: torch.LongTensor = None
        self._preds: Dict[str, torch.Tensor] = {}
        self._num_outputs_prepared: int = 0
        self._enabled = True

        if not (0.0 <= start < 1.0):
            raise ValueError(
                f"Expected `start` to be between 0.0 and 1.0, but got {start}."
            )
        if not (start <= stop <= 1.0):
            raise ValueError(
                f"Expected `stop` to be between {start} and 1.0, but got {stop}."
            )

        if self._input_predictions is None or not isinstance(self._input_predictions, list):
            raise ValueError(
                "`_input_predictions` must be a list of required prediction names for the guidance technique."
            )

    def force_disable(self):
        self._enabled = False
    
    def force_enable(self):
        self._enabled = True
    
    def set_state(self, step: int, num_inference_steps: int, timestep: torch.LongTensor) -> None:
        self._step = step
        self._num_inference_steps = num_inference_steps
        self._timestep = timestep
        self._preds = {}
        self._num_outputs_prepared = 0

    def prepare_models(self, denoiser: torch.nn.Module) -> None:
        """
        Prepares the models for the guidance technique on a given batch of data. This method should be overridden in
        subclasses to implement specific model preparation logic.
        """
        pass
    
    def prepare_inputs(self, denoiser: torch.nn.Module, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
        raise NotImplementedError("BaseGuidance::prepare_inputs must be implemented in subclasses.")

    def prepare_outputs(self, denoiser: torch.nn.Module, pred: torch.Tensor) -> None:
        raise NotImplementedError("BaseGuidance::prepare_outputs must be implemented in subclasses.")

    def __call__(self, **kwargs) -> Any:
        if len(kwargs) != self.num_conditions:
            raise ValueError(
                f"Expected {self.num_conditions} arguments, but got {len(kwargs)}. Please provide the correct number of arguments."
            )
        return self.forward(**kwargs)

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("BaseGuidance::forward must be implemented in subclasses.")

    @property
    def is_conditional(self) -> bool:
        raise NotImplementedError("BaseGuidance::is_conditional must be implemented in subclasses.")
    
    @property
    def is_unconditional(self) -> bool:
        return not self.is_conditional
    
    @property
    def num_conditions(self) -> int:
        raise NotImplementedError("BaseGuidance::num_conditions must be implemented in subclasses.")

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


def _default_prepare_inputs(denoiser: torch.nn.Module, num_conditions: int, *args: Union[Tuple[torch.Tensor], List[torch.Tensor]]) -> Tuple[List[torch.Tensor], ...]:
    """
    Prepares the inputs for the denoiser by ensuring that the conditional and unconditional inputs are correctly
    prepared based on required number of conditions. This function is used in the `prepare_inputs` method of the
    `BaseGuidance` class.

    Either tensors or tuples/lists of tensors can be provided. If a tuple/list is provided, it should contain two elements:
    - The first element is the conditional input.
    - The second element is the unconditional input or None.
    
    If only the conditional input is provided, it will be repeated for all batches.
    
    If both conditional and unconditional inputs are provided, they are alternated as batches of data.
    """
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
                # Alternating conditional and unconditional inputs as batches
                inputs = [arg[i % 2] for i in range(num_conditions)]
                list_of_inputs.append(inputs)
        else:
            raise ValueError(
                f"Expected a tensor, tuple, or list, but got {type(arg)} for argument {arg}. Please provide a tensor, tuple, or list."
            )
    return tuple(list_of_inputs)
