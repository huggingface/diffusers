# coding=utf-8
# Copyright 2025 HuggingFace Inc.
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


import torch

from diffusers import (
    ControlNetModel,
)

from ..testing_utils import (
    enable_full_determinism,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestControlNetModelSingleFile(SingleFileModelTesterMixin):
    model_class = ControlNetModel
    ckpt_path = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"
    repo_id = "lllyasviel/control_v11p_sd15_canny"

    def test_single_file_arguments(self):
        model_default = self.model_class.from_single_file(self.ckpt_path)

        assert model_default.config.upcast_attention is False
        assert model_default.dtype == torch.float32

        torch_dtype = torch.float16
        upcast_attention = True

        model = self.model_class.from_single_file(
            self.ckpt_path,
            upcast_attention=upcast_attention,
            torch_dtype=torch_dtype,
        )
        assert model.config.upcast_attention == upcast_attention
        assert model.dtype == torch_dtype
