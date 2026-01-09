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


from diffusers import (
    Lumina2Transformer2DModel,
)

from ..testing_utils import (
    enable_full_determinism,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestLumina2Transformer2DModelSingleFile(SingleFileModelTesterMixin):
    model_class = Lumina2Transformer2DModel
    ckpt_path = "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors"
    alternate_keys_ckpt_paths = [
        "https://huggingface.co/Comfy-Org/Lumina_Image_2.0_Repackaged/blob/main/split_files/diffusion_models/lumina_2_model_bf16.safetensors"
    ]

    repo_id = "Alpha-VLLM/Lumina-Image-2.0"
    subfolder = "transformer"
