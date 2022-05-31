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

# helpers functions

# NOTE: the following file is completely copied from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/master/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
from ...configuration_utils import PretrainedConfig


class UNetConfig(PretrainedConfig):
    model_type = "unet"

    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 4, 8),
        init_dim=None,
        out_dim=None,
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        learned_variance=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_mults = dim_mults
        self.init_dim = init_dim
        self.out_dim = out_dim
        self.channels = channels
        self.with_time_emb = with_time_emb
        self.resnet_block_groups = resnet_block_groups
        self.learned_variance = learned_variance
