# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
""" ConfigMixinuration base class and utilities."""


import copy
import inspect
import json
import os
import re
from typing import Any, Dict, Tuple, Union

from requests import HTTPError
from transformers.utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    EntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
)

from . import __version__


logger = logging.get_logger(__name__)

_re_configuration_file = re.compile(r"config\.(.*)\.json")


class ConfigMixin:
    r"""
    Base class for all configuration classes. Handles a few parameters common to all models' configurations as well as
    methods for loading/downloading/saving configurations.

    """
    config_name = None

    def register(self, **kwargs):
        if self.config_name is None:
            raise NotImplementedError(f"Make sure that {self.__class__} has defined a class name `config_name`")
        kwargs["_class_name"] = self.__class__.__name__
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

        if not hasattr(self, "_dict_to_save"):
            self._dict_to_save = {}

        self._dict_to_save.update(kwargs)

    def save_config(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~ConfigMixin.from_config`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we save using the predefined names, we can load using `from_config`
        output_config_file = os.path.join(save_directory, self.config_name)

        self.to_json_file(output_config_file)
        logger.info(f"ConfigMixinuration saved in {output_config_file}")

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        user_agent = {"file_type": "config"}

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            configuration_file = cls.config_name

            if os.path.isdir(pretrained_model_name_or_path):
                config_file = os.path.join(pretrained_model_name_or_path, configuration_file)
            else:
                config_file = hf_bucket_url(
                    pretrained_model_name_or_path, filename=configuration_file, revision=revision, mirror=None
                )

        try:
            # Load from URL or cache if already cached
            resolved_config_file = cached_path(
                config_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on "
                "'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having "
                "permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass "
                "`use_auth_token=True`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
                f"model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for "
                "available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {configuration_file}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in"
                f" the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory"
                f" containing a {configuration_file} file.\nCheckout your internet connection or see how to run the"
                " library in offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a {configuration_file} file"
            )

        try:
            # Load config dict
            config_dict = cls._dict_from_json_file(resolved_config_file)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if resolved_config_file == config_file:
            logger.info(f"loading configuration file {config_file}")
        else:
            logger.info(f"loading configuration file {config_file} from cache at {resolved_config_file}")

        expected_keys = set(dict(inspect.signature(cls.__init__).parameters).keys())
        expected_keys.remove("self")

        for key in expected_keys:
            if key in kwargs:
                # overwrite key
                config_dict[key] = kwargs.pop(key)

        passed_keys = set(config_dict.keys())

        unused_kwargs = kwargs
        for key in passed_keys - expected_keys:
            unused_kwargs[key] = config_dict.pop(key)

        if len(expected_keys - passed_keys) > 0:
            logger.warn(
                f"{expected_keys - passed_keys} was not found in config. Values will be initialized to default values."
            )

        return config_dict, unused_kwargs

    @classmethod
    def from_config(cls, pretrained_model_name_or_path: Union[str, os.PathLike], return_unused_kwargs=False, **kwargs):
        config_dict, unused_kwargs = cls.get_config_dict(
            pretrained_model_name_or_path=pretrained_model_name_or_path, **kwargs
        )

        model = cls(**config_dict)

        if return_unused_kwargs:
            return model, unused_kwargs
        else:
            return model

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Diffusion version when serializing the model
        output["diffusers_version"] = __version__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self._dict_to_save
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
