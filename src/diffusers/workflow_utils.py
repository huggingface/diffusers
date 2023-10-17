# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Module for managing workflows."""
import os
from typing import Dict, List, Optional, Union

import numpy as np
import PIL
import torch

from .configuration_utils import ConfigMixin, FrozenDict
from .utils import PushToHubMixin, logging
from .utils.constants import WORKFLOW_NAME


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_NON_CALL_ARGUMENTS = ["lora", "is_torch_tensor_present", "_name_or_path", "_class_name", "_diffusers_version"]
_ALLOWED_PATTERNS = r"^[\w\s.,!?@#$%^&*()_+-=<>[\]{}|\\;:'\"/]*$"


def populate_workflow_from_pipeline(
    argument_names: List[str], call_arg_values: Dict, lora_info: Optional[Dict], pipeline_name_or_path: str
) -> Dict:
    r"""Populates the call arguments and (optional) LoRA information in a dictionary.

    Args:
        argument_names (`List[str]`): List of function arguments.
        call_arg_values (`Dict`):
            Dictionary containing the arguments and their values from the current execution frame.
        lora_info (`Dict`, *optional*): Details of the LoRA checkpoints loaded in the pipeline.
        pipeline_name_or_path (`str`): Name or the local path to the pipeline that was used to generate the workflow.

    Returns:
        `Dict`: A dictionary containing the details of the pipeline call arguments and (optionally) LoRA checkpoint
        details.
    """
    workflow = Workflow()

    # Populate call arguments.
    call_arguments = {
        arg: call_arg_values[arg]
        for arg in argument_names
        if arg != "return_workflow"
        and not isinstance(call_arg_values[arg], (torch.Tensor, np.ndarray, PIL.Image.Image))
    }

    workflow.update(call_arguments)

    generator = workflow.pop("generator")
    if generator is not None:
        try:
            workflow.update({"seed": generator.initial_seed()})
        except Exception:
            workflow.update({"seed": None})

    # Handle the case when `load_lora_weights()` was called on a pipeline.
    if len(lora_info) > 0:
        workflow["lora"].update(lora_info)

    workflow["_name_or_path"] = pipeline_name_or_path

    return workflow


class Workflow(dict, ConfigMixin, PushToHubMixin):
    """Class sub-classing from native Python dict to have support for interacting with the Hub."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_name = WORKFLOW_NAME
        self._internal_dict = FrozenDict(dict(kwargs.items()))

    def save_workflow(self, **kwargs):
        self.save_config(**kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        self.save_workflow(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)
