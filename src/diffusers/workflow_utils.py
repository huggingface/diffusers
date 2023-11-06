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
import json
import os
from pathlib import PosixPath
from typing import Union

import numpy as np
from huggingface_hub import create_repo

from . import __version__
from .utils import PushToHubMixin, logging
from .utils.constants import WORKFLOW_NAME


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_NON_CALL_ARGUMENTS = {"_name_or_path", "scheduler_config", "_class_name", "_diffusers_version"}


class Workflow(dict, PushToHubMixin):
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

    # Copied from diffusers.configuration_utils.ConfigMixin.to_json_string
    def to_json_string(self) -> str:
        """
        Serializes the configuration instance to a JSON string.

        Returns:
            `str`:
                String containing all the attributes that make up the configuration instance in JSON format.
        """
        config_dict = self._internal_dict if hasattr(self, "_internal_dict") else {}
        config_dict["_class_name"] = self.__class__.__name__
        config_dict["_diffusers_version"] = __version__

        def to_json_saveable(value):
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, PosixPath):
                value = str(value)
            return value

        config_dict = {k: to_json_saveable(v) for k, v in config_dict.items()}
        # Don't save "_ignore_files" or "_use_default_values"
        config_dict.pop("_ignore_files", None)
        config_dict.pop("_use_default_values", None)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def save_workflow(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        filename: str = WORKFLOW_NAME,
        **kwargs,
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
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self.config_name = filename

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        output_config_file = os.path.join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            private = kwargs.pop("private", False)
            create_pr = kwargs.pop("create_pr", False)
            token = kwargs.pop("token", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = create_repo(repo_id, exist_ok=True, private=private, token=token).repo_id

            self._upload_folder(
                save_directory,
                repo_id,
                token=token,
                commit_message=commit_message,
                create_pr=create_pr,
            )

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
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self.save_workflow(
            save_directory=save_directory,
            push_to_hub=push_to_hub,
            filename=filename,
            **kwargs,
        )
