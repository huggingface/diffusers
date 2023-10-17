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
from typing import Dict, List, Optional

import torch

from .configuration_utils import ConfigMixin, FrozenDict
from .utils import PushToHubMixin, constants, logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def populate_workflow_from_pipeline(
    argument_names: List[str],
    call_arg_values: Dict,
    lora_info: Optional[Dict],
):
    r"""Populates the call arguments and LoRA information in a dictionary.

    Args:
        argument_names (`List[str]`): List of function arguments.
        call_arg_values (`Dict`):
            Dictionary containing the arguments and their values from the current execution frame.
        lora_info (`Dict`, *optional*): Details of the LoRA checkpoints loaded in the pipeline.
    """
    workflow = {}

    # Populate call arguments.
    call_arguments = {
        arg: call_arg_values[arg]
        for arg in argument_names
        if arg != "return_workflow" and not isinstance(call_arg_values[arg], torch.Tensor)
    }

    workflow.update({"call": call_arguments})

    generator = workflow["call"].pop("generator")
    if generator is not None:
        try:
            workflow["call"].update({"seed": generator.initial_seed()})
        except Exception:
            workflow["call"].update({"seed": None})

    # Handle the case for inputs that are of type torch tensors.
    is_torch_tensor_present = any(isinstance(call_arg_values[arg], torch.Tensor) for arg in argument_names)
    if is_torch_tensor_present:
        logger.warning(
            "`torch.Tensor`s detected in the call argument values. They won't be made a part of the workflow."
        )

    # Handle the case when `load_lora_weights()` was called on a pipeline.
    if len(lora_info) > 0:
        workflow["lora"].update(lora_info)

    # Make it shareable.
    workflow = Workflow(workflow)
    return workflow


class Workflow(ConfigMixin, PushToHubMixin):
    """Base class for managing workflows."""

    def __init__(self, workflow: Dict):
        self._internal_dict = FrozenDict(workflow)
        self.config_name = constants.WORKFLOW_NAME

    def save_workflow(self, **kwargs):
        self.save_config(**kwargs)
