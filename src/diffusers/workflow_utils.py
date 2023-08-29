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
from typing import Dict, List

import torch

from .configuration_utils import ConfigMixin, FrozenDict
from .utils import PushToHubMixin


WORKFLOW_NAME = "diffusion_workflow.json"


def populate_workflow_from_pipeline(argument_names: List[str], call_arg_values: Dict, pipeline_components: Dict):
    r"""Populates the pipeline components' configurations and the call arguments in a dictionary.

    Args:
        argument_names (`List[str]`): List of function arguments.
        call_arg_values (`Dict`):
            Dictionary containing the arguments and their values from the current execution frame.
        pipeline_components (`Dict`): Components of the underlying pipeline.
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

    # Populate component configs.
    workflow.update(
        {
            "components": {
                component_name: component.config
                if isinstance(component.config, FrozenDict)
                else component.config.to_dict()
                for component_name, component in pipeline_components.items()
                if hasattr(component, "config")
            }
        }
    )

    # Make it shareable.
    workflow = Workflow(workflow)
    return workflow


class Workflow(ConfigMixin, PushToHubMixin):
    """Base class for managing workflows."""

    def __init__(self, workflow: Dict):
        self._internal_dict = FrozenDict(workflow)
        self.config_name = WORKFLOW_NAME

    def save_workflow(self, **kwargs):
        self.save_config(**kwargs)
