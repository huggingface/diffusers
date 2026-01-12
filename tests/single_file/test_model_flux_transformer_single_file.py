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

import gc

from diffusers import (
    FluxTransformer2DModel,
)

from ..testing_utils import (
    backend_empty_cache,
    enable_full_determinism,
    torch_device,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestFluxTransformer2DModelSingleFile(SingleFileModelTesterMixin):
    model_class = FluxTransformer2DModel
    ckpt_path = "https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/flux1-dev.safetensors"
    alternate_keys_ckpt_paths = ["https://huggingface.co/Comfy-Org/flux1-dev/blob/main/flux1-dev-fp8.safetensors"]

    repo_id = "black-forest-labs/FLUX.1-dev"
    subfolder = "transformer"

    def test_device_map_cuda(self):
        backend_empty_cache(torch_device)
        model = self.model_class.from_single_file(self.ckpt_path, device_map="cuda")

        del model
        gc.collect()
        backend_empty_cache(torch_device)
