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
    WanTransformer3DModel,
)

from ..testing_utils import (
    enable_full_determinism,
    require_big_accelerator,
)
from .single_file_testing_utils import SingleFileModelTesterMixin


enable_full_determinism()


class TestWanTransformer3DModelText2VideoSingleFile(SingleFileModelTesterMixin):
    model_class = WanTransformer3DModel
    ckpt_path = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors"
    repo_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    subfolder = "transformer"


@require_big_accelerator
class TestWanTransformer3DModelImage2VideoSingleFile(SingleFileModelTesterMixin):
    model_class = WanTransformer3DModel
    ckpt_path = "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/blob/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors"
    repo_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    torch_dtype = torch.float8_e4m3fn
    subfolder = "transformer"
