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
    AutoencoderKLWan,
)

from ..testing_utils import (
    enable_full_determinism,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestAutoencoderKLWanSingleFile(SingleFileModelTesterMixin):
    model_class = AutoencoderKLWan
    ckpt_path = (
        "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/vae/wan_2.1_vae.safetensors"
    )
    repo_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    subfolder = "vae"
