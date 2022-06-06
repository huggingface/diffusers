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

import importlib
import os
from typing import Optional, Union

# CHANGE to diffusers.utils
from transformers.utils import logging

from .configuration_utils import Config


INDEX_FILE = "diffusion_model.pt"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
        "GaussianDDPMScheduler": ["save_config", "from_config"],
    },
    "transformers": {
        "PreTrainedModel": ["save_pretrained", "from_pretrained"],
    },
}


class DiffusionPipeline(Config):

    config_name = "model_index.json"

    def __init__(self, **kwargs):
        for name, module in kwargs.items():
            # retrive library
            library = module.__module__.split(".")[0]
            # retrive class_name
            class_name = module.__class__.__name__

            # save model index config
            self.register(**{name: (library, class_name)})

            # set models
            setattr(self, name, module)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        self.save_config(save_directory)

        model_index_dict = self._dict_to_save
        model_index_dict.pop("_class_name")

        for name, (library_name, class_name) in self._dict_to_save.items():
            importable_classes = LOADABLE_CLASSES[library_name]

            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name)
            class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

            save_method_name = None
            for class_name, class_candidate in class_candidates.items():
                if issubclass(class_obj, class_candidate):
                    save_method_name = importable_classes[class_name][0]

            save_method = getattr(getattr(self, name), save_method_name)
            save_method(os.path.join(save_directory, name))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs):
        # use snapshot download here to get it working from from_pretrained
        config_dict, _ = cls.get_config_dict(pretrained_model_name_or_path)

        init_kwargs = {}

        for name, (library_name, class_name) in config_dict.items():
            importable_classes = LOADABLE_CLASSES[library_name]

            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name)
            class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

            load_method_name = None
            for class_name, class_candidate in class_candidates.items():
                if issubclass(class_obj, class_candidate):
                    load_method_name = importable_classes[class_name][1]

            load_method = getattr(class_obj, load_method_name)

            loaded_sub_model = load_method(os.path.join(pretrained_model_name_or_path, name))

            init_kwargs[name] = loaded_sub_model

        model = cls(**init_kwargs)
        return model
