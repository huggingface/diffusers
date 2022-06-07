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
        "GaussianDDPMScheduler": ["save_config", "from_config"],
    },
    "transformers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
    },
}


class DiffusionPipeline(ConfigMixin):

    config_name = "model_index.json"

    def register_modules(self, **kwargs):
        for name, module in kwargs.items():
            # retrive library
            library = module.__module__.split(".")[0]
            # retrive class_name
            class_name = module.__class__.__name__

            register_dict = {name: (library, class_name)}
            

            # save model index config
            self.register(**register_dict)

            # set models
            setattr(self, name, module)
        
        register_dict = {"_module" : self.__module__.split(".")[-1] + ".py"}
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
        
        module = config_dict["_module"]
        class_name_ = config_dict["_class_name"]
        
        if class_name_ == cls.__name__:
            pipeline_class = cls
        else:
            pipeline_class = get_class_from_dynamic_module(cached_folder, module, class_name_, cached_folder)
        

        init_dict, _ = pipeline_class.extract_init_dict(config_dict, **kwargs)

        init_kwargs = {}

        for name, (library_name, class_name) in init_dict.items():
            importable_classes = LOADABLE_CLASSES[library_name]

            if library_name == module:
                # TODO(Suraj)
                pass

            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name)
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
