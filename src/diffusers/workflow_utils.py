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
from typing import Callable, Dict, List, Union

import numpy as np
import torch

from .configuration_utils import ConfigMixin
from .utils import PushToHubMixin, logging
from .utils.constants import WORKFLOW_NAME


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_NON_CALL_ARGUMENTS = ["lora", "is_torch_tensor_present", "_name_or_path", "_class_name", "_diffusers_version"]
_ALLOWED_PATTERNS = r"^[\w\s.,!?@#$%^&*()_+-=<>[\]{}|\\;:'\"/]*$"


def populate_workflow_from_pipeline(argument_names: List[str], call_arg_values: Dict, pipeline_obj) -> Dict:
    r"""Populates the call arguments and (optional) LoRA information in a dictionary.

    Args:
        argument_names (`List[str]`): List of function arguments.
        call_arg_values (`Dict`):
            Dictionary containing the arguments and their values from the current execution frame.
        pipeline_obj (`DiffusionPipeline`): The pipeline object.

    Returns:
        `Dict`: A dictionary containing the details of the pipeline call arguments and (optionally) LoRA checkpoint
        details.
    """
    # A `Workflow` object is an extended Python dictionary. So, all regular dictionary methods
    # apply to it.
    workflow = Workflow()

    # Populate call arguments.
    call_arguments = {
        arg: call_arg_values[arg]
        for arg in argument_names
        if arg != "return_workflow"
        and "image" not in arg
        and not isinstance(call_arg_values[arg], (torch.Tensor, np.ndarray, Callable))
    }
    workflow.update(call_arguments)
    print(f"call_arguments: {call_arguments}")
    print(f"workflow: {workflow}")

    # Handle generator device and seed. 
    print(workflow["generator"])
    generator = workflow.pop("generator")
    print(f"From workflow_utils: {generator.initial_seed()}")
    if generator is not None:
        workflow.update({"generator_seed": generator.initial_seed()})
        workflow.update({"generator_device": generator.device})
    else:
        workflow.update({"generator_seed": None})
        workflow.update({"generator_device": "cpu"})

    # Handle pipeline-level things.
    pipeline_config_name_or_path = (
        pipeline_obj.config._name_or_path if hasattr(pipeline_obj.config, "_name_or_path") else None
    )
    workflow["_name_or_path"] = pipeline_config_name_or_path
    workflow["scheduler_config"] = pipeline_obj.scheduler.config

    return workflow


class Workflow(dict, ConfigMixin, PushToHubMixin):
    """Class sub-classing from native Python dict to have support for interacting with the Hub."""

    config_name = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config_name = WORKFLOW_NAME
        self._internal_dict = {}

    def __setitem__(self, __key, __value):
        self._internal_dict[__key] = __value
        return super().__setitem__(__key, __value)

    def update(self, __m, **kwargs):
        self._internal_dict.update(__m, **kwargs)
        super().update(__m, **kwargs)

    def pop(self, __key):
        self._internal_dict.pop(__key)
        super().pop(__key)

    def save_workflow(
        self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, filename: str = WORKFLOW_NAME
    ):
        """
        Saves a workflow to a directory.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the workflow JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            filename (`str`, *optional*, defaults to `workflow.json`):
                Optional filename to use to serialize the workflow JSON.
        """
        self.config_name = filename
        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        filename: str = WORKFLOW_NAME,
        **kwargs,
    ):
        """
        Saves a workflow to a directory. This internally calls [`Workflow.save_workflow`], This method exists to have
        feature parity with [`PushToHubMixin.push_to_hub`].

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the workflow JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            filename (`str`, *optional*, defaults to `workflow.json`):
                Optional filename to use to serialize the workflow JSON.
        """
        self.save_workflow(save_directory=save_directory, push_to_hub=push_to_hub, filename=filename)
