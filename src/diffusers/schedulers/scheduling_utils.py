# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import numpy as np
import torch


SCHEDULER_CONFIG_NAME = "scheduler_config.json"


class SchedulerMixin:

    config_name = SCHEDULER_CONFIG_NAME

    def set_format(self, tensor_format="pt"):
        self.tensor_format = tensor_format
        if tensor_format == "pt":
            for key, value in vars(self).items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, torch.from_numpy(value))

        return self

    def clip(self, tensor, min_value=None, max_value=None):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.clip(tensor, min_value, max_value)
        elif tensor_format == "pt":
            return torch.clamp(tensor, min_value, max_value)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")

    def log(self, tensor):
        tensor_format = getattr(self, "tensor_format", "pt")

        if tensor_format == "np":
            return np.log(tensor)
        elif tensor_format == "pt":
            return torch.log(tensor)

        raise ValueError(f"`self.tensor_format`: {self.tensor_format} is not valid.")
