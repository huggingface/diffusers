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

from huggingface_hub import snapshot_download

# CHANGE to diffusers.utils
from transformers.utils import logging

from .configuration_utils import ConfigMixin
from .dynamic_modules_utils import get_class_from_dynamic_module


INDEX_FILE = "diffusion_model.pt"


logger = logging.get_logger(__name__)


LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "CLIPTextModel": ["save_pretrained", "from_pretrained"],  # TODO (Anton): move to transformers
        "GaussianDDPMScheduler": ["save_config", "from_config"],
        "ClassifierFreeGuidanceScheduler": ["save_config", "from_config"],
        "GlideDDIMScheduler": ["save_config", "from_config"],
    },
    "transformers": {
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


class DiffusionPipeline(ConfigMixin):

    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrive library
            library = module.__module__.split(".")[0]
            # if library is not in LOADABLE_CLASSES, then it is a custom module
            if library not in LOADABLE_CLASSES:
                library = module.__module__.split(".")[-1]

            # retrive class_name
            class_name = module.__class__.__name__

            register_dict = {name: (library, class_name)}

            # save model index config
            self.register(**register_dict)

            # set models
            setattr(self, name, module)

        register_dict = {"_module": self.__module__.split(".")[-1] + ".py"}
        self.register(**register_dict)

    def save_pretrained(self, save_directory: Union[str, os.PathLike]):
        self.save_config(save_directory)

        model_index_dict = self._dict_to_save
        model_index_dict.pop("_class_name")
        model_index_dict.pop("_module")

        for name, (library_name, class_name) in self._dict_to_save.items():
            importable_classes = LOADABLE_CLASSES[library_name]

            # TODO: Suraj
            if library_name == self.__module__:
                library_name = self

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
        if not os.path.isdir(pretrained_model_name_or_path):
            cached_folder = snapshot_download(pretrained_model_name_or_path)
        else:
            cached_folder = pretrained_model_name_or_path

        config_dict = cls.get_config_dict(cached_folder)

        module_candidate = config_dict["_module"]
        module_candidate_name = module_candidate.replace(".py", "")

        # if we load from explicit class, let's use it
        if cls != DiffusionPipeline:
            pipeline_class = cls
        else:
            # else we need to load the correct module from the Hub
            class_name_ = config_dict["_class_name"]
            module = module_candidate
            pipeline_class = get_class_from_dynamic_module(cached_folder, module, class_name_, cached_folder)

        init_dict, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        init_kwargs = {}

        for name, (library_name, class_name) in init_dict.items():
            
            # if the model is not in diffusers or transformers, we need to load it from the hub
            # assumes that it's a subclass of ModelMixin
            if library_name == module_candidate_name:
                class_obj = get_class_from_dynamic_module(cached_folder, module, class_name, cached_folder)
                # since it's not from a library, we need to check class candidates for all importable classes
                importable_classes = ALL_IMPORTABLE_CLASSES
                class_candidates = {c: class_obj for c in ALL_IMPORTABLE_CLASSES.keys()}
            else:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, class_name)
                importable_classes = LOADABLE_CLASSES[library_name]
                class_candidates = {c: getattr(library, c) for c in importable_classes.keys()}

            load_method_name = None
            for class_name, class_candidate in class_candidates.items():
                if issubclass(class_obj, class_candidate):
                    load_method_name = importable_classes[class_name][1]

            load_method = getattr(class_obj, load_method_name)

            if os.path.isdir(os.path.join(cached_folder, name)):
                loaded_sub_model = load_method(os.path.join(cached_folder, name))
            else:
                loaded_sub_model = load_method(cached_folder)

            init_kwargs[name] = loaded_sub_model  # UNet(...), # DiffusionSchedule(...)

        model = pipeline_class(**init_kwargs)
        return model
